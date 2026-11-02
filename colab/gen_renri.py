import sys,os,random,time,numpy as np,soundfile as sf
sys.path.insert(0,'/home/claude/h'); os.environ.setdefault('RSPEED','20')
import ds
RD=ds.RD
def load_dict(fn):
    out=[]
    for l in open(RD+fn,encoding='utf-8'):
        p=l.rstrip('\n').split('\t')
        if len(p)==2: out.append(p[1].replace('|',' ').split())
    return out
JP=load_dict('dsdict_JPN.txt'); EN=load_dict('dsdict_ENG.txt')
EN=[w for w in EN if 2<=len(w)<=6][:40000]
ph=set(l.strip() for l in open(RD+'0910_multi_langs_V5_2nd_fhyy_ds1000_1.phonemes.txt') if l.strip())
VOW=set('A1 I1 U1 E1 O1 aa ae ah ao2 eh er2 ih iy uh uw ow ay aw ey OY2 A I U E O'.split())
FPS=44100/512
def phrase(R):
    seq=[]; lang=R.random()
    n=R.randint(6,14)
    while len(seq)<n*2:
        w=R.choice(JP if lang<0.5 else EN) if R.random()<0.85 else R.choice(JP if lang>=0.5 else EN)
        w=[x for x in w if x in ph]
        if w: seq+=w
    return seq
def clip(R,seed):
    seq=phrase(R); mode=R.random()  # <0.45 sung notes, <0.8 speech/rap contour, else mixed
    base=R.uniform(57,72)  # midi center (A3..C5)
    dur=[R.randint(15,30)]; names=['SP']
    for p in seq:
        names.append(p); dur.append(R.randint(9,40) if p in VOW else R.randint(4,9))
        if R.random()<0.06: names.append('SP'); dur.append(R.randint(6,20))
    names.append('SP'); dur.append(15)
    T=sum(dur); f0=np.zeros(T,np.float32); t=0
    scale=[0,2,3,5,7,8,10,12,14,15]
    cur=base
    for nm,d in zip(names,dur):
        if mode<0.45: target=base+R.choice(scale)-5 if nm in VOW else cur
        else: target=base+R.uniform(-1.5,2.5) if nm in VOW else cur
        for k in range(d):
            x=k/max(1,d-1)
            if mode<0.45:
                m=cur+(target-cur)*min(1,k/4); 
                if d>25 and k>12: m+=0.25*np.sin(2*np.pi*5.5*(k-12)/FPS)
            else:
                m=target+ (1.2-1.8*x)*R.uniform(0.3,1.0) - 2.5*(t+k)/max(1,T)  # speech: fall in syllable + declination
            f0[t+k]=440*2**((m-69)/12)
        cur=target; t+=d
    a=ds.renri({},f0,names,dur)
    return a
if __name__=='__main__':
    out=sys.argv[1]; n0,n1=int(sys.argv[2]),int(sys.argv[3]); os.makedirs(out,exist_ok=True); tot=0; t0=time.time()
    for i in range(n0,n1):
        R=random.Random(i*7919+13); a=clip(R,i); a=a/ (np.abs(a).max()+1e-6)*0.8
        sf.write(f'{out}/renri_{i:04d}.flac',a,44100); tot+=len(a)/44100
        print(i,'%.1fs total %.0fs elapsed %.0fs'%(len(a)/44100,tot,time.time()-t0),flush=True)
