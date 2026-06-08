python scripts/gmr_to_npz.py --input_file /home/eric/Data/LAFAN1/lafan1-gmr-tg --output_name tg-lfan1 --headless

python scripts/replay_gmr.py --motion_file motion_data/tg-lfan1.npz

python scripts/rsl_rl/train.py --task=Tracking-Flat-TG-v0 \
--motion_file motion_data/tg-lfan1.npz \
--headless --logger tensorboard --run_name tg-lfan1

python scripts/rsl_rl/play.py --task=Tracking-Flat-TG-v0 --num_envs=1 \
--load_run=".*tg-lfan1.*" --checkpoint="model_.*.pt"

python scripts/gmr_to_npz.py --input_file /media/eai/data1/wbt/lafan1-dex/w1s2.pkl --output_name tg-w1s2 --frame_range 1 1000 --headless

python scripts/gmr_to_npz.py --input_file /media/eai/data1/wbt/lafan1-dex/d1s2.pkl --output_name tg-d1s2 --frame_range 1 700 --headless


python scripts/replay_gmr.py --motion_file motion_data/tg-w2s4.npz

python scripts/rsl_rl/train.py --task=Tracking-Flat-TG-v0 \
--motion_file motion_data/tg-w2s4.npz \
--headless --logger tensorboard --run_name tg-w2s4


python scripts/rsl_rl/train.py --task=Tracking-Flat-TG-v0 \
--motion_file motion_data/tg-d1s2.npz \
--headless --logger tensorboard --run_name tg-d1s2

python scripts/rsl_rl/play.py --task=Tracking-Flat-TG-v0 --num_envs=1 \
--load_run=".*tg-d1s2.*" --checkpoint="model_.*.pt"
