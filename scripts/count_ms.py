import glob, os, sys, json
import numpy as np, cv2
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, "/home/archryzen/yolo")
from detect import YoloNCNN
ROOT="/home/archryzen/bccd/datasets/merged"; CONFS=np.round(np.arange(0.05,0.96,0.05),2)
NAMES=["RBC","WBC","Platelets"]
def gt(split):
    g=[]
    for p in sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg")):
        b=os.path.splitext(os.path.basename(p))[0]; c=np.zeros(3,int)
        lab=f"{ROOT}/labels/{split}/{b}.txt"
        if os.path.exists(lab):
            for line in open(lab):
                t=line.split()
                if len(t)==5: c[int(t[0])]+=1
        g.append(c)
    return np.stack(g)
def run(split, models, iou):
    out=[]
    for p in sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg")):
        im=cv2.imread(p); h,w=im.shape[:2]; bl,sl,ll=[],[],[]
        for m in models:
            d=m(im,0.02,iou)
            if not len(d):
                bl.append(np.zeros((0,4))); sl.append(np.zeros((0,))); ll.append(np.zeros((0,))); continue
            bl.append(d[:,:4]/np.array([w,h,w,h])); sl.append(d[:,4]); ll.append(d[:,5].astype(int))
        bb,ss,ll2=weighted_boxes_fusion(bl,sl,ll,iou_thr=0.55,skip_box_thr=0.0)
        out.append((ss,ll2) if len(bb) else (np.zeros(0),np.zeros(0)))
    return out
def counts(dets,conf):
    c=np.zeros((len(dets),3),int)
    for i,(s,l) in enumerate(dets):
        if len(s):
            k=l[s>=conf]
            for j in range(3): c[i,j]=int((k==j).sum())
    return c
m640=YoloNCNN("bccd_v26n_640_ncnn_model")
m960=YoloNCNN("bccd_v26n_960_ncnn_model")
Gv,Gt=gt("val"),gt("test")
dv=run("val",[m640,m960],0.5); dt=run("test",[m640,m960],0.5)
mae,c=min(((np.abs(counts(dv,x)-Gv).mean(),x) for x in CONFS),key=lambda t:t[0])
P=counts(dt,c); E=np.abs(P-Gt)
print(f"multi-scale 640+960 WBF: val conf {c:.2f} (val MAE {mae:.2f})")
for k in range(3):
    print(f"  {NAMES[k]:10s} MAE {E[:,k].mean():5.2f} bias {(P[:,k]-Gt[:,k]).mean():+.2f} total {P[:,k].sum()}/{Gt[:,k].sum()}")
print(f"  all-class MAE {E.sum(1).mean():.2f}")
