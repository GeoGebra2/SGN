import os
import re
import cv2
import sys
import pickle
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown

DET_CONFIG = '../checkpoints/rtmdet_tiny_8xb32-300e_coco.py'
DET_CHECKPOINT = '../checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

POSE_CONFIG = '../checkpoints/rtmpose-m_8xb256-420e_coco-256x192.py'
POSE_CHECKPOINT = '../checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'


COCO_TO_NTU = {
    0: 3,
    5: 4,
    6: 8,
    7: 5,
    8: 9,
    9: 6,
    10: 10,
    11: 12,
    12: 16,
    13: 13,
    14: 17,
    15: 14,
    16: 18,
}

def parse_name(name):
    m = re.match(r'.*S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)', name)
    if not m:
        return None
    s, c, p, r, a = map(int, m.groups())
    return dict(setup=s, camera=c, subject=p, repeat=r, action=a)

def coco17_to_ntu25(seq17):
    T = seq17.shape[0]
    ntu = np.zeros((T, 25, 3), dtype=np.float32)
    for coco_id, ntu_id in COCO_TO_NTU.items():
        ntu[:, ntu_id, :] = seq17[:, coco_id, :]
    ntu[:, 1, :] = (ntu[:, 12, :] + ntu[:, 16, :]) / 2
    ntu[:, 2, :] = (ntu[:, 4, :] + ntu[:, 8, :]) / 2
    ntu[:, 0, :] = ntu[:, 1, :]
    return ntu

def extract_pose_sequence(video_path, det_model, pose_model):
    cap = cv2.VideoCapture(video_path)
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        init_default_scope('mmdet')
        det_result = inference_detector(det_model, frame)
        bboxes = det_result.pred_instances.bboxes
        scores = det_result.pred_instances.scores
        if len(bboxes) == 0:
            seq.append(np.zeros((17, 3), dtype=np.float32))
            continue
        idx = scores.argmax().item()
        bbox = bboxes[idx].cpu().numpy()
        init_default_scope('mmpose')
        pose_results = inference_topdown(pose_model, frame, [bbox])
        if len(pose_results) == 0 or len(pose_results[0].pred_instances.keypoints) == 0:
            seq.append(np.zeros((17, 3), dtype=np.float32))
            continue
        kpts = pose_results[0].pred_instances.keypoints[0]
        kpts_score = pose_results[0].pred_instances.keypoint_scores[0]
        frame_kps = np.concatenate([kpts, kpts_score[:, None]], axis=1).astype(np.float32)
        seq.append(frame_kps)
    cap.release()
    if len(seq) == 0:
        return np.zeros((1, 17, 3), dtype=np.float32)
    return np.stack(seq)

def _pad_to_max(seqs, max_frame):
    if len(seqs) == 0:
        return np.zeros((0, max_frame, 150), dtype=np.float32)
    Tm = max_frame
    out = np.zeros((len(seqs), Tm, 150), dtype=np.float32)
    for i, s in enumerate(seqs):
        t = min(s.shape[0], Tm)
        out[i, :t] = s[:t]
    return out

def _one_hot(labels, num_classes):
    m = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, l in enumerate(labels):
        m[i, l] = 1
    return m

def build_dataset(video_dir, out_folder, device, benchmark, max_frame, num_joint, max_body_true, det_cfg, det_ckpt, pose_cfg, pose_ckpt):
    os.makedirs(out_folder, exist_ok=True)
    det_model = init_detector(det_cfg, det_ckpt, device=device)
    pose_model = init_model(pose_cfg, pose_ckpt, device=device)
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    videos.sort()
    train_id_data, test_id_data = [], []
    train_id_labels, test_id_labels = [], []
    train_act_data, test_act_data = [], []
    train_act_labels, test_act_labels = [], []
    for vid in tqdm(videos):
        meta = parse_name(os.path.splitext(vid)[0])
        if meta is None:
            continue
        video_path = os.path.join(video_dir, vid)
        seq17 = extract_pose_sequence(video_path, det_model, pose_model)
        seq25 = coco17_to_ntu25(seq17)
        T = seq25.shape[0]
        seq_vec = seq25.reshape(T, -1)
        seq150 = np.zeros((T, 150), dtype=np.float32)
        seq150[:, :75] = seq_vec
        if benchmark == 'xview':
            is_train = meta['camera'] in [2, 3]
        else:
            is_train = meta['subject'] in [1, 2, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        if is_train:
            train_id_data.append(seq150)
            train_id_labels.append(meta['subject'] - 1)
            train_act_data.append(seq150)
            train_act_labels.append(meta['action'] - 1)
        else:
            test_id_data.append(seq150)
            test_id_labels.append(meta['subject'] - 1)
            test_act_data.append(seq150)
            test_act_labels.append(meta['action'] - 1)
    train_id_x = _pad_to_max(train_id_data, max_frame)
    test_id_x = _pad_to_max(test_id_data, max_frame)
    train_act_x = _pad_to_max(train_act_data, max_frame)
    test_act_x = _pad_to_max(test_act_data, max_frame)
    if train_id_x.shape[0] == 0 and test_id_x.shape[0] == 0:
        return
    idx = np.arange(train_id_x.shape[0])
    if idx.size > 0:
        np.random.seed(10000)
        np.random.shuffle(idx)
    val_num = int(np.ceil(0.05 * idx.size)) if idx.size > 0 else 0
    val_idx = idx[:val_num]
    tr_idx = idx[val_num:]
    tr_id_x = train_id_x[tr_idx] if tr_idx.size > 0 else np.zeros((0, max_frame, 150), dtype=np.float32)
    va_id_x = train_id_x[val_idx] if val_idx.size > 0 else np.zeros((0, max_frame, 150), dtype=np.float32)
    tr_id_y = [train_id_labels[i] for i in tr_idx]
    va_id_y = [train_id_labels[i] for i in val_idx]
    te_id_y = test_id_labels
    tr_act_x = train_act_x[tr_idx] if tr_idx.size > 0 else np.zeros((0, max_frame, 150), dtype=np.float32)
    va_act_x = train_act_x[val_idx] if val_idx.size > 0 else np.zeros((0, max_frame, 150), dtype=np.float32)
    tr_act_y = [train_act_labels[i] for i in tr_idx]
    va_act_y = [train_act_labels[i] for i in val_idx]
    te_act_y = test_act_labels
    bench = 'CV' if benchmark == 'xview' else 'CS'
    id_file = os.path.join(out_folder, f'NTU_ID_{bench}.h5')
    with h5py.File(id_file, 'w') as f:
        f.create_dataset('x', data=tr_id_x)
        f.create_dataset('y', data=_one_hot(tr_id_y, 40))
        f.create_dataset('valid_x', data=va_id_x)
        f.create_dataset('valid_y', data=_one_hot(va_id_y, 40))
        f.create_dataset('test_x', data=test_id_x)
        f.create_dataset('test_y', data=_one_hot(te_id_y, 40))
    act_file = os.path.join(out_folder, f'NTU_{bench}.h5')
    with h5py.File(act_file, 'w') as f:
        f.create_dataset('x', data=tr_act_x)
        f.create_dataset('y', data=_one_hot(tr_act_y, 60))
        f.create_dataset('valid_x', data=va_act_x)
        f.create_dataset('valid_y', data=_one_hot(va_act_y, 60))
        f.create_dataset('test_x', data=train_act_x if test_act_x.shape[0] == 0 else test_act_x)
        f.create_dataset('test_y', data=_one_hot(tr_act_y if test_act_x.shape[0] == 0 else te_act_y, 60))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='./Recordings')
    parser.add_argument('--out_folder', type=str, default='./ntu')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--benchmark', type=str, choices=['xview', 'xsub'], default='xview')
    parser.add_argument('--max_frame', type=int, default=300)
    parser.add_argument('--num_joint', type=int, default=25)
    parser.add_argument('--max_body_true', type=int, default=2)
    parser.add_argument('--det_config', type=str, default=DET_CONFIG)
    parser.add_argument('--det_checkpoint', type=str, default=DET_CHECKPOINT)
    parser.add_argument('--pose_config', type=str, default=POSE_CONFIG)
    parser.add_argument('--pose_checkpoint', type=str, default=POSE_CHECKPOINT)
    args = parser.parse_args()
    build_dataset(
        args.video_dir,
        args.out_folder,
        args.device,
        args.benchmark,
        args.max_frame,
        args.num_joint,
        args.max_body_true,
        args.det_config,
        args.det_checkpoint,
        args.pose_config,
        args.pose_checkpoint,
    )

if __name__ == '__main__':
    main()
