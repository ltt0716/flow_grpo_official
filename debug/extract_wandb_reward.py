#!/usr/bin/env python
"""
extract_wandb_reward.py —— 从 wandb 离线 run 里抽 reward 历史(无需联网)

offline run 被 SIGKILL 杀掉时往往没落 wandb-summary.json,但 reward 历史都在
二进制 .wandb 文件里。本脚本直接解析它,导出 debug/wandb_reward.csv 供分析。

用法:
    python debug/extract_wandb_reward.py            # 自动扫所有 offline-run
    python debug/extract_wandb_reward.py wandb/offline-run-XXXX   # 指定某个
然后:
    git add -f debug/wandb_reward.csv && git commit -m reward && git push
"""
import sys, os, glob, json, csv

def find_runs():
    if len(sys.argv) > 1:
        return [sys.argv[1]]
    return sorted(glob.glob("wandb/offline-run-*")) + sorted(glob.glob("wandb/run-*"))

def parse_wandb_file(path):
    """用 wandb 内部 DataStore 解析 .wandb,返回 history dict 列表。"""
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    rows = []
    dumped = False
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            if not dumped:
                # 打印一条原始 history 记录,确认字段结构(key vs nested_key)
                print("=== 样例 history 记录(原始 proto)===")
                print(str(rec.history)[:800])
                print("=== /样例 ===")
                dumped = True
            row = {}
            for item in rec.history.item:
                # 列名:优先 key,空则用 nested_key 拼接
                k = item.key or ".".join(item.nested_key) or "?"
                try:
                    row[k] = json.loads(item.value_json)
                except Exception:
                    row[k] = item.value_json
            rows.append(row)
    return rows

def main():
    runs = find_runs()
    if not runs:
        print("没找到 wandb offline run 目录"); return
    all_rows = []
    for run in runs:
        # 先打印 summary(若有)
        summ = os.path.join(run, "files", "wandb-summary.json")
        if os.path.exists(summ):
            print(f"\n--- {run} summary ---")
            print(open(summ).read())
        wfiles = glob.glob(os.path.join(run, "*.wandb"))
        for wf in wfiles:
            try:
                rows = parse_wandb_file(wf)
                print(f"--- {wf}: 解析到 {len(rows)} 条 history ---")
                all_rows.extend(rows)
            except Exception as e:
                print(f"✗ 解析 {wf} 失败: {type(e).__name__}: {e}")

    if not all_rows:
        print("没解析到 history(可能 wandb 版本不兼容,把上面输出贴回来)"); return

    # 收集所有出现过的列,优先 reward/epoch/kl
    keys = []
    for r in all_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    pref = [k for k in keys if any(t in k.lower() for t in ("epoch", "step", "reward", "kl"))]
    cols = pref + [k for k in keys if k not in pref]

    out = "debug/wandb_reward.csv"
    os.makedirs("debug", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n✅ 导出 {len(all_rows)} 行 -> {out}")
    # 末尾打印几条 reward 走势速览
    rk = next((k for k in cols if "reward" in k.lower() and "avg" in k.lower()), None) \
         or next((k for k in cols if "reward" in k.lower()), None)
    ek = next((k for k in cols if k.lower() in ("epoch", "_step", "step")), None)
    if rk:
        print(f"\nreward 列 = {rk}  (epoch 列 = {ek})  最近 10 条:")
        for r in all_rows[-10:]:
            print(f"  {ek}={r.get(ek)}  {rk}={r.get(rk)}")

if __name__ == "__main__":
    main()
