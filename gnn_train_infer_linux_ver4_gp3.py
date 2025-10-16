#!/usr/bin/env python3
# -*- coding: utf-8 -*-
DATA_DIR   = "/caedata2/WIASPH/pjh/python_test/inp_data/new"                 
RESULT_DIR = "/caedata2/WIASPH/pjh/python_test/inp_data/new/result_data_gp3"                 
INFER_DIR  = "/caedata2/WIASPH/pjh/python_test/inp_data/new/infer_data_gp3"                 
EPOCHS     = 150
LR         = 1e-3
DROPOUT    = 0.2
K_PARAM    = 1.0
L_PARAM    = 1.0
M_PARAM    = 1.0
SEED       = 20251016


import os, random, numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from sklearn.metrics import r2_score
import re
import sys, logging, builtins
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def init_logger(result_dir: Path, filename: str = "run.log", overwrite: bool = False):
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / filename
    file_mode = "w" if overwrite else "a"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode=file_mode, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    def _print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        logging.info(msg)
    builtins.print = _print
    logging.info(f"=== Logging to: {log_path} ===")
    return log_path
    
# === Progress helper (tqdm???덉쑝硫??ъ슜, ?놁쑝硫?媛꾨떒 異쒕젰) ===
try:
    from tqdm.auto import tqdm
    def progress(iterable, **kwargs):
        return tqdm(iterable, **kwargs)
except Exception:
    def progress(iterable, desc="", total=None):
        # fallback: tqdm ?놁쓣 ??媛꾨떒 濡쒓렇
        if total is None:
            total = len(iterable) if hasattr(iterable, "__len__") else None
        for i, item in enumerate(iterable, 1):
            if total:
                print(f"{desc} {i}/{total} ({int(i/total*100)}%)")
            else:
                print(f"{desc} {i}")
            yield item

def _has_torch_cluster_ops():
    try:
        from torch_geometric.nn import knn_graph, radius_graph  # noqa
        return True
    except Exception:
        return False

def safe_knn_graph(pos: torch.Tensor, k: int):
    if _has_torch_cluster_ops():
        from torch_geometric.nn import knn_graph
        return knn_graph(x=pos, k=k, batch=None, loop=False, num_workers=0)
    # Fallback (O(N^2))
    N = pos.size(0)
    with torch.no_grad():
        d = torch.cdist(pos, pos, p=2)
        d.fill_diagonal_(float('inf'))
        vals, idx = torch.topk(d, k, largest=False)
        src = torch.arange(N, device=pos.device).unsqueeze(1).expand(N, k)
        ei = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=0)
    return ei

def safe_radius_graph(pos: torch.Tensor, r: float, max_num_neighbors: int = 32):
    if _has_torch_cluster_ops():
        from torch_geometric.nn import radius_graph
        return radius_graph(x=pos, r=r, batch=None, loop=False,
                            max_num_neighbors=max_num_neighbors, num_workers=0)
    # Fallback (O(N^2))
    with torch.no_grad():
        d = torch.cdist(pos, pos, p=2)
        mask = (d <= r)
        mask.fill_diagonal_(False)
        idxs = mask.nonzero(as_tuple=False)   # [E,2] (src,row, tgt,col)
        src, tgt = idxs[:, 0], idxs[:, 1]
        order = torch.argsort(d[src, tgt])
        src, tgt = src[order], tgt[order]
        counts = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        keep = torch.zeros_like(src, dtype=torch.bool)
        for i in range(src.numel()):
            t = int(tgt[i].item())
            if counts[t] < max_num_neighbors:
                keep[i] = True
                counts[t] += 1
        src, tgt = src[keep], tgt[keep]
        return torch.stack([src, tgt], dim=0)

# ========= Model =========
class SparseGraphConvolution(nn.Module):
    """
    edge_attr = [dist, sim, dirx, diry, dirz] (dim=5)
    message = (XW)[src] * edge_enc(edge_attr) * sim * dist
    aggregate = mean over incoming edges
    """
    def __init__(self, in_feats, out_feats, edge_dim=6, activation=F.relu):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_feats, out_feats))
        self.bias   = nn.Parameter(torch.empty(out_feats))
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(out_feats + 3, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X, edge_index, edge_attr):
        support = X @ self.weight
        src, tgt = edge_index
        dist   = edge_attr[:, 0:1]
        sim    = edge_attr[:, 1:2]
        dirvec = edge_attr[:, 2:5]

        edge_feat = self.edge_encoder(edge_attr)
        m = support[src] * edge_feat
        m = m * sim * dist

        m = self.scale_mlp(torch.cat([m, dirvec], dim=1))  # [E,F]
        N = X.size(0)
        out = torch.zeros(N, m.size(1), device=X.device, dtype=m.dtype)
        out.index_add_(0, tgt, m)
        deg = torch.bincount(tgt, minlength=N).clamp_min(1).unsqueeze(1).to(m.dtype)
        out = out / deg
        out = out + self.bias
        return self.activation(out) if self.activation else out

class SimpleGNN(nn.Module):
    def __init__(self, in_dim=4, hid_dim=32, out_dim=1, edge_dim=6, dropout=0.5):
        super().__init__()
        self.gc1  = SparseGraphConvolution(in_dim, hid_dim, edge_dim, F.relu)
        self.ln1  = nn.LayerNorm(hid_dim)
        self.res1 = nn.Linear(in_dim, hid_dim)
        self.gc2  = SparseGraphConvolution(hid_dim, hid_dim, edge_dim, F.relu)
        self.ln2  = nn.LayerNorm(hid_dim)
        self.res2 = nn.Linear(hid_dim, hid_dim)
        self.gc3  = SparseGraphConvolution(hid_dim, out_dim, edge_dim, activation=None)
        self.dropout = dropout

    def forward(self, X, edge_index, edge_attr):
        h1 = self.gc1(X, edge_index, edge_attr)
        h1 = self.ln1(h1 + self.res1(X))
        h1 = F.relu(h1); h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.gc2(h1, edge_index, edge_attr)
        h2 = self.ln2(h2 + self.res2(h1))
        h2 = F.relu(h2); h2 = F.dropout(h2, p=self.dropout, training=self.training)
        return self.gc3(h2, edge_index, edge_attr)  # (N,1)

# ========= Geometry / Graph utils =========
def _expand_element_to_tets(idx, etype):
    """idx: list[int] (0-based node indices) → list[tuple4] of tetra corner indices"""
    if etype in ("C3D4", "C3D10"):  # tetra (10노드는 앞 4개만 코너)
        return [tuple(idx[:4])]
    if etype in ("C3D5",):  # pyramid 5 → 2 tets
        i = idx[:5]
        return [(i[0], i[1], i[2], i[4]),
                (i[0], i[2], i[3], i[4])]
    if etype in ("C3D6",):  # wedge/prism 6 → 3 tets
        i = idx[:6]
        return [(i[0], i[1], i[2], i[3]),
                (i[3], i[4], i[5], i[1]),
                (i[1], i[2], i[3], i[5])]
    if etype in ("C3D8", "C3D20"):  # hexa 8 → 5 tets
        i = idx[:8]
        return [(i[0], i[1], i[3], i[4]),
                (i[1], i[2], i[3], i[6]),
                (i[1], i[3], i[4], i[6]),
                (i[1], i[5], i[4], i[6]),
                (i[3], i[4], i[6], i[7])]
    # 기타 타입은 스킵
    return []


def compute_node_volumes_fast(pos: torch.Tensor,
                              elements: list,   # [{"type": "...", "conn": [node_ids...]}]
                              id2idx: dict,
                              cache_key: str = None,
                              cache_dir: Path = None):
    """
    초고속 체적 → 노드 분배.
    pos: [N,3] (CPU/GPU 모두 OK; GPU면 더 빠름)
    반환: node_volume [N] (float32)
    """
    device = pos.device
    N = pos.size(0)

    # ---- 캐시 사용 (선택) ----
    if cache_key and cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}_nodevol.pt"
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location=device)
                if isinstance(cached, torch.Tensor) and cached.numel() == N:
                    return cached.to(device)
            except:  # 캐시 파손 시 무시
                pass

    # 1) 모든 요소를 테트라 인덱스로 확장 (0-based)
    tet_nodes = []
    for e in elements:
        etype = e["type"]
        # 연결 노드 id → index
        try:
            idx = [id2idx[n] for n in e["conn"] if n in id2idx]
        except KeyError:
            continue
        if len(idx) < 4:
            continue
        tets = _expand_element_to_tets(idx, etype)
        tet_nodes.extend(tets)

    if not tet_nodes:
        return torch.zeros(N, dtype=pos.dtype, device=device)

    tet_idx = torch.tensor(tet_nodes, dtype=torch.long, device=device)  # [T,4]

    # 2) 테트라 체적 배치 계산
    # P: [T,4,3]; V = |det([p1-p0, p2-p0, p3-p0])| / 6
    P = pos[tet_idx]                               # [T,4,3]
    A = P[:, 1] - P[:, 0]                          # [T,3]
    B = P[:, 2] - P[:, 0]                          # [T,3]
    C = P[:, 3] - P[:, 0]                          # [T,3]
    M = torch.stack([A, B, C], dim=2)              # [T,3,3]
    V = torch.abs(torch.linalg.det(M)) / 6.0       # [T]

    # 3) 노드별 분배 (각 테트라 4등분)
    #   scatter_add with flattened indices
    share = (V / 4.0).unsqueeze(1).repeat(1, 4)    # [T,4]
    flat_idx = tet_idx.reshape(-1)                 # [4T]
    flat_val = share.reshape(-1)                   # [4T]
    node_vol = torch.zeros(N, dtype=pos.dtype, device=device)
    node_vol.index_add_(0, flat_idx, flat_val)

    # ---- 캐시 저장 (선택) ----
    if cache_key and cache_dir:
        try:
            torch.save(node_vol.detach().to("cpu"), cache_path)
        except:
            pass

    return node_vol

##################################################################
def tet_volume(p0, p1, p2, p3):
    # | (p1-p0) · ((p2-p0) x (p3-p0)) | / 6
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    return torch.abs(torch.linalg.det(torch.stack([v1, v2, v3], dim=1))) / 6.0

def hexa_to_tets(idx8):
    """
    C3D8 코너 8개(0-based) → 5개 테트라 분해(일반적 분해 중 하나).
    Abaqus C3D8 순서 가정: 1-2-3-4(바닥), 5-6-7-8(상부) → 0..7
    """
    i = idx8
    return [
        (i[0], i[1], i[3], i[4]),
        (i[1], i[2], i[3], i[6]),
        (i[1], i[3], i[4], i[6]),
        (i[1], i[5], i[4], i[6]),
        (i[3], i[4], i[6], i[7]),
    ]

def wedge_to_tets(idx6):
    """C3D6 (삼각기둥) 6개 코너 → 3개 테트라 분해"""
    i = idx6
    return [
        (i[0], i[1], i[2], i[3]),
        (i[3], i[4], i[5], i[1]),
        (i[1], i[2], i[3], i[5]),
    ]

def pyramid_to_tets(idx5):
    """C3D5 (피라미드) 5개 코너(0..3: 바닥 4: 정점) → 2개 테트라"""
    i = idx5
    return [
        (i[0], i[1], i[2], i[4]),
        (i[0], i[2], i[3], i[4]),
    ]
def compute_node_volumes(pos: torch.Tensor, elements: list, id2idx: dict):
    N = pos.size(0)
    node_vol = torch.zeros(N, device=pos.device, dtype=pos.dtype)

    for e in elements:
        etype = e["type"]
        # 코너 노드만 취득 (쿼드릭 요소는 앞쪽 코너 4/5/6/8개 사용)
        conn_ids = e["conn"]
        if etype in ("C3D10",):   # 테트라 10노드
            corners = conn_ids[:4]
        elif etype in ("C3D20",): # 헥사 20노드
            corners = conn_ids[:8]
        else:
            corners = conn_ids

        # id → index
        try:
            idx = torch.tensor([id2idx[nid] for nid in corners], device=pos.device, dtype=torch.long)
        except KeyError:
            continue

        # 요소 별 체적
        vol = 0.0
        if etype in ("C3D4", "C3D10"):  # tetra
            if idx.numel() >= 4:
                p = pos[idx[:4]]
                vol = float(tet_volume(p[0], p[1], p[2], p[3]))
        elif etype in ("C3D6",):  # wedge/prism
            if idx.numel() >= 6:
                tets = wedge_to_tets(idx[:6].tolist())
                vol = 0.0
                for a,b,c,d in tets:
                    p = pos[torch.tensor([a,b,c,d])]
                    vol += float(tet_volume(p[0], p[1], p[2], p[3]))
        elif etype in ("C3D5",):  # pyramid
            if idx.numel() >= 5:
                tets = pyramid_to_tets(idx[:5].tolist())
                vol = 0.0
                for a,b,c,d in tets:
                    p = pos[torch.tensor([a,b,c,d])]
                    vol += float(tet_volume(p[0], p[1], p[2], p[3]))
        elif etype in ("C3D8", "C3D20"):  # hexa
            if idx.numel() >= 8:
                tets = hexa_to_tets(idx[:8].tolist())
                vol = 0.0
                for a,b,c,d in tets:
                    p = pos[torch.tensor([a,b,c,d])]
                    vol += float(tet_volume(p[0], p[1], p[2], p[3]))
        else:
            # 다른 타입은 우선 스킵(필요하면 추가)
            continue

        if vol <= 0.0:
            continue

        k = idx.numel()
        share = vol / k
        node_vol.index_add_(0, idx, torch.full((k,), share, device=pos.device, dtype=pos.dtype))

    return node_vol
    
# ========= Geometry / Graph utils =========
def parse_inp_nodes_and_solid_elements(inp_file_path: Path):
    nodes = {}
    elements = []
    cur_elem_type = None
    in_node = False
    in_elem = False

    with open(inp_file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            up = line.upper()

            if up.startswith("*NODE"):
                in_node, in_elem = True, False
                continue

            if up.startswith("*ELEMENT"):
                in_node, in_elem = False, True
                # TYPE= 파싱
                m = re.search(r"TYPE\s*=\s*([A-Za-z0-9]+)", up)
                cur_elem_type = m.group(1) if m else None
                continue

            if line.startswith("*"):  # 다른 섹션 시작
                in_node = in_elem = False
                continue

            if in_node:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    try:
                        nid = int(parts[0]); x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                        nodes[nid] = [x, y, z]
                    except: pass

            if in_elem:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and cur_elem_type is not None:
                    # 첫 값은 element id, 나머지가 노드 라벨들
                    try:
                        conn = [int(p) for p in parts[1:] if p and re.match(r"^-?\d+$", p)]
                        if len(conn) >= 4:  # 체적 요소는 최소 4개 노드
                            elements.append({"type": cur_elem_type.upper(), "conn": conn})
                    except: pass

    return nodes, elements
    
def convert_inp_to_graph(inp_file_path: Path, csv_path: Path,                                   
                         k_pa: float, l_pa: float, m_pa: float, device: torch.device):  
    # ==== Edge switches ====
    USE_MESH=True; USE_KNN=True; USE_RADIUS=False
    kp = 20  # 또는 8/16 그리드로 튜닝

    # 1) INP 파싱(노드 + 체적 요소)  ← elements_list 변수 **생성**                             
    nodes_dict, elements_list = parse_inp_nodes_and_solid_elements(inp_file_path)               
    # elements_list: [{"type": "C3D8", "conn": [node_ids...]}]                                  
                                                                                                
    # 노드 인덱스 매핑                                                                          
    sorted_node_ids = sorted(nodes_dict.keys())                                                 
    id2idx = {nid: i for i, nid in enumerate(sorted_node_ids)}                                  
                                                                                                
    # 좌표 텐서                                                                                 
    pos = torch.tensor([nodes_dict[n] for n in sorted_node_ids],                                
                       dtype=torch.float32, device=device)  # [N,3]                             
    N = pos.size(0)                                                                             
                                                                                                
    # 2) 좌표 정규화                                                                            
    pmin = pos.min(dim=0, keepdim=True).values                                                  
    pmax = pos.max(dim=0, keepdim=True).values                                                  
    pos_n = (pos - pmin) / (pmax - pmin + 1e-8)                                                 
                                                                                                
    # 3) 초고속 체적 -> 노드 분배  ← **elements_list 사용**                                    
    cache_dir = Path(RESULT_DIR) / "cache"                                                      
    cache_key = Path(inp_file_path).stem  # 예: assembly_005_merged_renum                       
    node_vol = compute_node_volumes_fast(                                                       
        pos=pos,                                                                                
        elements=elements_list,                                                                 
        id2idx=id2idx,                                                                          
        cache_key=cache_key,                                                                    
        cache_dir=cache_dir                                                                     
    ).unsqueeze(1)  # [N,1]                                                                     
                                                                                                
    # min-max 정규화                                                                            
    vmin = node_vol.min(dim=0, keepdim=True).values                                             
    vmax = node_vol.max(dim=0, keepdim=True).values                                             
    vol_norm = (node_vol - vmin) / (vmax - vmin + 1e-8)                                         
                                                                                                
    # 4) (선택) 물리 메타 피처를 붙이고 싶다면 여기서 호출                                      
    # meta = parse_inp_metadata(inp_file_path)                                                  
    # node_feats_dict, edge_aug_fn, K_type = build_physical_features(pos, sorted_node_ids, meta)
                                                                                                
    # 5) 노드 특성 X (현재는 pos_norm + volume_norm)                                            
    X = torch.cat([pos_n, vol_norm], dim=1)  # [N,4]                                            
                                                                                                
    # 6) 엣지 구성: Mesh-edges + KNN + Radius 하이브리드
    raw = np.sort(np.array([k_pa, l_pa, m_pa], dtype=np.float32))
    small, mid, large = raw[0], raw[1], raw[2]
    kp = int(np.clip(small * 2, 8, 24))
    lp = float(np.clip((small + large) / 2.0, 0.02, 0.25))
    mp = float(np.clip((small + mid + large) / 3.0, 0.03, 0.20))

    # 6-1) Mesh edges (요소 연결 기반)
    # elements_list: [{"type": "...", "conn": [node_ids...]}]
    mesh_pairs = []
    for e in elements_list:
        conn_ids = [nid for nid in e["conn"] if nid in id2idx]
        if len(conn_ids) < 2:
            continue
        idxs = [id2idx[nid] for nid in conn_ids]
        # 요소 내 모든 노드쌍(무방향) 추가
        for i in range(len(idxs)):
            ui = idxs[i]
            for j in range(i + 1, len(idxs)):
                vj = idxs[j]
                mesh_pairs.append((ui, vj))
                mesh_pairs.append((vj, ui))  # 양방향

    if len(mesh_pairs) > 0:
        ei_mesh = torch.tensor(mesh_pairs, device=device, dtype=torch.long).T  # [2, E_mesh*2]
        # 중복 제거
        vals_mesh = torch.ones(ei_mesh.size(1), device=device)
        ei_mesh = torch.sparse_coo_tensor(ei_mesh, vals_mesh, (N, N)).coalesce().indices()
    else:
        ei_mesh = torch.empty((2, 0), dtype=torch.long, device=device)

    # 6-2) 좌표 기반 KNN/Radius
    ei_knn = safe_knn_graph(pos_n, k=kp)  # [2, E_knn]
    ei_rad = safe_radius_graph(pos_n, r=lp, max_num_neighbors=kp)  # [2, E_rad]

    ei_parts = []
    # 1) Mesh edges
    if USE_MESH:
        ei_parts.append(ei_mesh)  # (2, E_mesh) ? 요소에서 만든 엣지

    # 2) KNN edges
    if USE_KNN and (kp is not None) and (kp >= 1):
        ei_knn = safe_knn_graph(pos_n, k=kp)   # (2, E_knn)
        ei_parts.append(ei_knn)

    # 3) Radius edges
    if USE_RADIUS and (lp is not None) and (lp > 0.0):
        ei_rad = safe_radius_graph(pos_n, r=lp, max_num_neighbors=max_nn)
        ei_parts.append(ei_rad)

    # 4) 합치고 중복 제거
    if len(ei_parts) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        ei_all = torch.cat(ei_parts, dim=1)
        vals   = torch.ones(ei_all.size(1), device=device)
        edge_index = torch.sparse_coo_tensor(ei_all, vals, (N, N)).coalesce().indices()

##    # 6-3) 합치고 최종 중복 제거
##    ei_all = torch.cat([ei_mesh, ei_knn, ei_rad], dim=1)
##    vals_all = torch.ones(ei_all.size(1), device=device)
##    edge_index = torch.sparse_coo_tensor(ei_all, vals_all, (N, N)).coalesce().indices()  # [2, E]

    # 6-4) is_mesh 플래그 생성 (메시 엣지인지 여부를 엣지 특성에 추가)
    #   - 파이썬 set으로 membership 체크 (E가 매우 큰 경우 필요시 더 최적화)
    mesh_undirected = set()
    # undirected 판별을 위해 (min,max) 형태로 보관
    if ei_mesh.numel() > 0:
        u_m, v_m = ei_mesh[0].tolist(), ei_mesh[1].tolist()
        for a, b in zip(u_m, v_m):
            mesh_undirected.add((a, b) if a <= b else (b, a))

    u_all = edge_index[0].tolist()
    v_all = edge_index[1].tolist()
    is_mesh_list = [1.0 if ((a, b) if a <= b else (b, a)) in mesh_undirected else 0.0
                    for a, b in zip(u_all, v_all)]
    is_mesh_flag = torch.tensor(is_mesh_list, device=device, dtype=torch.float32).unsqueeze(1)  # [E,1]

    # 6-5) 기본 edge_attr (거리/유사도/방향) + is_mesh
    src, tgt = edge_index
    dvec = pos_n[tgt] - pos_n[src]
    dist = torch.norm(dvec, p=2, dim=1, keepdim=True)
    sim  = torch.exp(-dist / (mp + 1e-12))
    dirn = dvec / (dvec.norm(dim=1, keepdim=True) + 1e-8)
    base_attr = torch.cat([dist, sim, dirn], dim=1)  # [E,5]

    # per-column min-max 정규화
    cmin = base_attr.min(0, keepdim=True).values
    cmax = base_attr.max(0, keepdim=True).values
    base_attr = (base_attr - cmin) / (cmax - cmin + 1e-6)

    # is_mesh는 그대로 스칼라(0/1)로 붙임 (정규화 X)
    edge_attr = torch.cat([base_attr, is_mesh_flag], dim=1)  # [E,6] (dist,sim,dx,dy,dz,is_mesh)
                                                                                         
                                                                                                
    # 7) 라벨 y (그대로)
    df = pd.read_csv(csv_path).set_index('Node Label').sort_index()
    y = torch.tensor(
        df.reindex(sorted_node_ids)['maxPrincipal'].fillna(0.0).values,
        dtype=torch.float32, device=device
    )

    return {
        "x": X, "edge_index": edge_index, "edge_attr": edge_attr,
        "y": y, "node_labels": sorted_node_ids
    }
                                                         
# ========= Train / Eval / Infer =========
def save_model_state_dict(model: nn.Module, save_dir: Path, filename: str = "best_model.pth"):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_dir / filename))

def load_model_state_dict(model_cls, load_path: Path, device, **kwargs):
    model = model_cls(**kwargs).to(device)
    state = torch.load(str(load_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def evaluate_on_graphs(model, graphs, y_mean, y_std, device):
    if len(graphs) == 0:
        print("[Eval] No graphs to evaluate (skipping).")
        return float("nan"), float("nan"), float("nan"), torch.tensor([]), torch.tensor([])

    total_mse_orig, total_mse_z = 0.0, 0.0
    ys_orig, ps_orig = [], []

    for g in graphs.values():
        x, ei, ea, y = g["x"], g["edge_index"], g["edge_attr"], g["y"]
        pred_z = model(x, ei, ea).squeeze(1)               # z-스케일 예측
        pred_orig = pred_z * y_std + y_mean                # 역정규화
        # 누적
        total_mse_orig += F.mse_loss(pred_orig, y).item()
        y_z   = (y - y_mean) / y_std
        total_mse_z    += F.mse_loss(pred_z, y_z).item()

        ys_orig.append(y.detach().cpu())
        ps_orig.append(pred_orig.detach().cpu())

    y_true = torch.cat(ys_orig); y_pred = torch.cat(ps_orig)

    try:
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true.numpy(), y_pred.numpy())
        mae = mean_absolute_error(y_true.numpy(), y_pred.numpy())
    except Exception:
        r2, mae = float("nan"), float("nan")

    mse_orig = total_mse_orig / max(1, len(graphs))
    mse_z    = total_mse_z    / max(1, len(graphs))
    print(f"[Eval] Test z-MSE={mse_z:.4f} | MSE(orig)={mse_orig:.4f} | MAE(orig)={mae:.4f} | R²={r2:.4f}")
    return mse_orig, r2, y_true, y_pred  # ← 기존 인터페이스 유지 (필요시 mse_z도 함께 반환 가능)
    
def train_gnn(train_graphs, val_graphs, test_graphs, result_dir: Path,
              in_dim=4, hid_dim=32, out_dim=1, edge_dim=5,
              epochs=100, lr=1e-3, dropout=0.5, seed=20250922, device=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGNN(in_dim, hid_dim, out_dim, edge_dim, dropout=dropout).to(device)
    optimz = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses   = []
    
    # z-score from train
    all_train_y = torch.cat([g["y"] for g in train_graphs.values()]).to(device)
    y_train = torch.cat([g["y"] for g in train_graphs.values()]).to(device)
    y_log   = torch.log1p(y_train.clamp_min(0))
    y_mean, y_std = all_train_y.mean(), all_train_y.std().clamp_min(1e-6)

    best_val, best_state = float('inf'), None
    train_items = list(train_graphs.items())
    val_items   = list(val_graphs.items())

    for ep in range(1, epochs+1):
        model.train(); total = 0.0
        for _, g in train_items:
            x, ei, ea, y = g["x"], g["edge_index"], g["edge_attr"], g["y"]
            y_scaled = ((y.unsqueeze(1)) - y_mean) / y_std
            optimz.zero_grad()
#########################################################
#########################################################
            out = model(x, ei, ea)
            loss = F.mse_loss(out, y_scaled)
#########################################################
#########################################################
##            y_true = graph.y.unsqueeze(1).to(device)
##            y_z    = (torch.log1p(y_true.clamp_min(0)) - y_mean) / y_std
##            pred_z = model(X, graph.edge_index, graph.edge_attr)
#########################################################
#########################################################
##            loss = 0.7*F.mse_loss(pred_z, y_z) + 0.3*F.l1_loss(pred_z, y_z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimz.step()
            total += loss.item()
        tr = total / max(1, len(train_items))

        model.eval(); vtot = 0.0
        with torch.no_grad():
            for _, g in val_items:
                x, ei, ea, y = g["x"], g["edge_index"], g["edge_attr"], g["y"]
                y_scaled = ((y.unsqueeze(1)) - y_mean) / y_std
                out = model(x, ei, ea)
                vtot += F.mse_loss(out, y_scaled).item()
        va = vtot / max(1, len(val_items))
        
        train_losses.append(tr)
        val_losses.append(va)
        
        if va < best_val:
            best_val = va
            best_state = model.state_dict()
            save_model_state_dict(model, result_dir, "best_model.pth")
        print(f"[{ep:03d}] train(zMSE)={tr:.4f}  val(zMSE)={va:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, test_r2, y_true, y_pred = evaluate_on_graphs(model, test_graphs, y_mean, y_std, device)
    print(f"Test MSE={test_mse:.4f}, R2={test_r2:.4f}")
    
    # --- save epoch-loss plot & csv ---                                     
    epochs_range = list(range(1, len(train_losses)+1))                       
    fig, ax = plt.subplots(figsize=(6,4), dpi=120)                           
    ax.plot(epochs_range, train_losses, label="Train z-MSE")                 
    ax.plot(epochs_range, val_losses,   label="Val z-MSE")                   
    ax.set_xlabel("Epoch")                                                   
    ax.set_ylabel("Loss (z-MSE)")                                            
    ax.set_title("Epoch vs. Loss")                                           
    ax.grid(True, alpha=0.3)                                                 
    ax.legend()                                                              
    (result_dir / "epoch_loss.png").parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()                                                       
    fig.savefig(result_dir / "epoch_loss.png")                               
    plt.close(fig)                                                           
                                                                                                                                   
    import pandas as pd                                                      
    df_loss = pd.DataFrame({                                                 
        "epoch": epochs_range,                                               
        "train_zmse": train_losses,                                          
        "val_zmse": val_losses,                                              
    })                                                                       
    df_loss.to_csv(result_dir / "epoch_loss.csv", index=False)               
    print(f"[Plot] saved: {result_dir / 'epoch_loss.png'}")                  
    print(f"[CSV ] saved: {result_dir / 'epoch_loss.csv'}")                  
    
    return model, (y_mean, y_std), (test_mse, test_r2, y_true, y_pred)

@torch.no_grad()
def infer_on_folder(model, y_stats, infer_dir: Path, out_dir: Path,
                    k_pa=6.0, l_pa=4.0, m_pa=5.5, device=None):
    """
    - 각 INP에 대해 예측 -> GT(maxPrincipal)과 조인 -> per-file CSV 저장
    - 모든 파일 합쳐서 합본 CSV 저장
    - 파일별/전체 MAE, RMSE, MSE, R2(가능시) 요약 저장
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    device = device or next(model.parameters()).device
    y_mean, y_std = y_stats
    out_dir.mkdir(parents=True, exist_ok=True)

    # 후보 INP 수집 (패턴 유틸 사용)
    all_files = sorted(os.listdir(infer_dir))
    inp_files = [fn for fn in all_files if fn.lower().endswith(".inp")]
    candidates = []
    for fn in inp_files:
        ok, prefix = match_inp_and_prefix(fn)
        if ok: candidates.append((fn, prefix))

    if len(candidates) == 0:
        print(f"[Infer] No .inp candidates found in {infer_dir}")
        return

    merged_rows = []
    metrics_rows = []
    skipped = 0

    for fn, prefix in progress(candidates, desc="[Infer] graphs", total=len(candidates)):
        inp = infer_dir / fn
        csv = csv_path_for_prefix(infer_dir, prefix)
        if not csv.exists():
            print(f"[skip] CSV not found: {csv}"); skipped += 1; continue

        print(f"[{prefix}] building graph...")
        g = convert_inp_to_graph(inp, csv, k_pa, l_pa, m_pa, device)

        # 예측 (역정규화)
        pred_z = model(g["x"], g["edge_index"], g["edge_attr"]).squeeze(1)
        pred   = (pred_z * y_std + y_mean).detach().cpu().numpy()
        node_labels = np.array(g["node_labels"])

        # GT 로드 (Node Label, maxPrincipal)
        df_gt = pd.read_csv(csv)
        if not {"Node Label", "maxPrincipal"}.issubset(df_gt.columns):
            print(f"[error] CSV columns missing in {csv.name}: need 'Node Label' and 'maxPrincipal'")
            skipped += 1
            continue

        df_gt = df_gt[["Node Label", "maxPrincipal"]].rename(
            columns={"Node Label": "node_label", "maxPrincipal": "gt"}
        )

        # 예측 DF
        df_pred = pd.DataFrame({"node_label": node_labels, "pred": pred})

        # 조인 (inner)
        df_join = df_pred.merge(df_gt, on="node_label", how="inner")
        if len(df_join) == 0:
            print(f"[warn] No overlapping node labels for {prefix}. Skipping metrics.")
            skipped += 1
            continue

        # 에러 컬럼
        df_join["err"]  = df_join["pred"] - df_join["gt"]
        df_join["abs"]  = df_join["err"].abs()
        df_join["sq"]   = df_join["err"] ** 2

        # Per-file 지표
        y_true = df_join["gt"].to_numpy()
        y_hat  = df_join["pred"].to_numpy()
        mae = float(mean_absolute_error(y_true, y_hat))
        mse = float(mean_squared_error(y_true, y_hat))
        rmse = float(np.sqrt(mse))
        try:
            r2 = float(r2_score(y_true, y_hat)) if len(df_join) >= 2 else float("nan")
        except Exception:
            r2 = float("nan")

        metrics_rows.append({
            "file": prefix,
            "n_nodes": int(len(df_join)),
            "MAE": mae,
            "RMSE": rmse,
            "MSE": mse,
            "R2": r2,
        })

        # 개별 파일 저장 (예측+GT+에러)
        out_csv = out_dir / f"{prefix}_pred_gt.csv"
        df_join[["node_label", "pred", "gt", "err", "abs", "sq"]].to_csv(out_csv, index=False)
        print(f"[{prefix}] saved -> {out_csv}")

        # 합본 수집
        df_join_ins = df_join.copy()
        df_join_ins.insert(0, "file", prefix)
        merged_rows.append(df_join_ins)

    # 합본/요약 저장
    if merged_rows:
        df_all = pd.concat(merged_rows, ignore_index=True)
        all_csv = out_dir / "infer_predictions_all_with_gt.csv"
        df_all.to_csv(all_csv, index=False)
        print(f"[Infer] merged {len(merged_rows)} files -> {all_csv}")
        print(df_all.head(5).to_string(index=False))

        # 파일별 지표 저장
        df_metrics = pd.DataFrame(metrics_rows)
        per_file_csv = out_dir / "infer_metrics_per_file.csv"
        df_metrics.to_csv(per_file_csv, index=False)
        print(f"[Infer] per-file metrics -> {per_file_csv}")

        # 전체 지표 (모든 노드 합침)
        y_true_all = df_all["gt"].to_numpy()
        y_hat_all  = df_all["pred"].to_numpy()
        mae_all = float(mean_absolute_error(y_true_all, y_hat_all))
        mse_all = float(mean_squared_error(y_true_all, y_hat_all))
        rmse_all = float(np.sqrt(mse_all))
        try:
            r2_all = float(r2_score(y_true_all, y_hat_all)) if len(df_all) >= 2 else float("nan")
        except Exception:
            r2_all = float("nan")

        df_overall = pd.DataFrame([{
            "files": len(merged_rows),
            "rows": len(df_all),
            "MAE": mae_all,
            "RMSE": rmse_all,
            "MSE": mse_all,
            "R2": r2_all,
        }])
        overall_csv = out_dir / "infer_metrics_overall.csv"
        df_overall.to_csv(overall_csv, index=False)
        print(f"[Infer] overall metrics -> {overall_csv}")
        print(df_overall.to_string(index=False))
    else:
        print(f"[Infer] No predictions written (skipped={skipped})")


# ========= Data IO =========
INP_PATTERNS = [
    re.compile(r"^(?P<prefix>.+)_stl\.inp$", re.IGNORECASE),
    re.compile(r"^(?P<prefix>.+)_merged_renum\.inp$", re.IGNORECASE),
]
def match_inp_and_prefix(filename: str):
    for pat in INP_PATTERNS:
        m = pat.match(filename)
        if m:
            return True, m.group("prefix")
    return False, None

def csv_path_for_prefix(folder: Path, prefix: str) -> Path:
    return folder / f"{prefix}_average_nodal_stress_new.csv"

def load_graphs_from_folder(data_path: Path, k_pa, l_pa, m_pa, device):
    graphs = {}
    all_files = sorted(os.listdir(data_path))
    inp_files = [fn for fn in all_files if fn.lower().endswith(".inp")]
    print(f"[Load] Found {len(inp_files)} .inp files in {data_path}")

    matched = []
    unmatched = []
    for fn in inp_files:
        ok, prefix = match_inp_and_prefix(fn)
        (matched if ok else unmatched).append((fn, prefix))
    print(f"[Load] Matched INP: {[f for f,_ in matched]}")
    if unmatched:
        print(f"[Load] Unmatched INP (ignored): {[f for f,_ in unmatched]}")

    for fn, prefix in progress(matched, desc="[Build] graphs", total=len(matched)):
        inp = data_path / fn
        csv = csv_path_for_prefix(data_path, prefix)
        if not csv.exists():
            print(f"[skip] CSV not found for {prefix}: {csv}")
            continue
        try:
            print(f"[{prefix}] parsing INP/CSV...")
            g = convert_inp_to_graph(inp, csv, k_pa, l_pa, m_pa, device)
            graphs[prefix] = g
            n_nodes = int(g["x"].shape[0]); n_edges = int(g["edge_index"].shape[1])
            print(f"[{prefix}] nodes={n_nodes}, edges={n_edges}")
        except Exception as e:
            print(f"[error] Failed for {prefix}: {type(e).__name__}: {e}")
    print(f"[Load] Built {len(graphs)} graphs total")
    return graphs

def split_graphs(graphs: dict, seed=20250922, train_ratio=0.7, val_ratio=0.15):
    keys = list(graphs.keys())
    rng = random.Random(seed); rng.shuffle(keys)
    n = len(keys)
    n_tr = int(train_ratio * n)
    n_va = int(val_ratio * n)
    train_keys = keys[:n_tr]
    val_keys   = keys[n_tr:n_tr+n_va]
    test_keys  = keys[n_tr+n_va:]
    return ({k: graphs[k] for k in train_keys},
            {k: graphs[k] for k in val_keys},
            {k: graphs[k] for k in test_keys})

# ========= Main =========
if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    data_dir   = Path(DATA_DIR)
    result_dir = Path(RESULT_DIR)
    infer_dir  = Path(INFER_DIR) if INFER_DIR else data_dir

    log_path = init_logger(result_dir, filename="run.log", overwrite=False)
    print(f"[Logger] writing to {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Config] DATA_DIR={data_dir}  RESULT_DIR={result_dir}  INFER_DIR={infer_dir}")
    print(f"[Config] EPOCHS={EPOCHS}  LR={LR}  DROPOUT={DROPOUT}  K={K_PARAM}  L={L_PARAM}  M={M_PARAM}")

    try:
        graphs = load_graphs_from_folder(data_dir, K_PARAM, L_PARAM, M_PARAM, device)
        if len(graphs) == 0:
            raise SystemExit(f"[Fatal] 0 graphs. "
                             f"Expect pairs like: assembly_442_merged_renum.inp + assembly_442_average_nodal_stress_new.csv")

        train_g, val_g, test_g = split_graphs(graphs, seed=SEED)

        model, y_stats, test_metrics = train_gnn(
            train_g, val_g, test_g, result_dir,
            in_dim=4, hid_dim=32, out_dim=1, edge_dim=6,
            epochs=EPOCHS, lr=LR, dropout=DROPOUT,
            seed=SEED, device=device
        )

        best_model_path = result_dir / "best_model.pth"
        loaded = load_model_state_dict(SimpleGNN, best_model_path, device,
                                       in_dim=4, hid_dim=32, out_dim=1, edge_dim=6, dropout=DROPOUT)
        infer_on_folder(loaded, y_stats, infer_dir, infer_dir,
                        k_pa=K_PARAM, l_pa=L_PARAM, m_pa=M_PARAM, device=device)
        print("=== Done ===")
    except Exception:
        logging.exception("Unhandled exception during run")
        raise
