"""Audit actual source presentation endpoints versus container-declared duration."""
import json,subprocess,hashlib,time
from pathlib import Path
import numpy as np
def main():
    O=Path('results/q1_deep_v6/media_clock');O.mkdir(exist_ok=True,parents=True);rows=[];start=time.perf_counter()
    for p in sorted(Path('results/q1/samples').glob('*.json')):
        m=json.loads(p.read_text());source=Path('/data')/m['source_path'];assert hashlib.sha256(source.read_bytes()).hexdigest()==m['source_sha256']
        a=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(source)]
        frames=json.loads(subprocess.check_output(a))['frames'];pts=np.array([float(f['best_effort_timestamp_time'])-m['container_origin_s'] for f in frames]);assert np.all(np.diff(pts)>0) and len(pts)==m['original_video_frames']
        lasts=frames[-1].get('pkt_duration_time');dt=float(lasts) if lasts is not None else float(np.median(np.diff(pts)));assert dt>0
        videoend=float(pts[-1]+dt);audioend=m['audio_start_s']+m['decoded_audio_duration_s'];observed=max(videoend,audioend);declared=m['duration_s']
        row=dict(stem=p.stem,sample_id=m['sample_id'],container_duration_s=declared,audio_start_s=m['audio_start_s'],audio_decoded_duration_s=m['decoded_audio_duration_s'],audio_end_s=audioend,video_first_pts_s=float(pts[0]),video_last_pts_s=float(pts[-1]),last_frame_duration_s=dt,last_frame_duration_source='pkt_duration_time' if lasts is not None else 'median_PTS_delta_fallback',video_presentation_end_s=videoend,observed_end_s=observed,empty_declared_tail_s=max(0,declared-observed),original_video_frames=len(pts),source_sha256=m['source_sha256'])
        rows.append(row)
    result=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),samples=len(rows),rows=rows,declared_total_s=sum(r['container_duration_s'] for r in rows),observed_end_total_s=sum(r['observed_end_s'] for r in rows),empty_tail_total_s=sum(r['empty_declared_tail_s'] for r in rows),over500ms_tail_samples=sum(r['empty_declared_tail_s']>.5 for r in rows),last_packet_duration_present=sum(r['last_frame_duration_source']=='pkt_duration_time' for r in rows),seconds=time.perf_counter()-start)
    (O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
