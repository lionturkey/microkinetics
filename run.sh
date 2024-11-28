python micro_runner.py frugal --reward frugal --num_timesteps 1000000
python micro_runner.py frugal --reward frugal --run_type load
python micro_runner.py frugal --reward frugal --run_type load --noise 0.01
python micro_runner.py frugal --reward frugal --run_type load --noise 0.05
python micro_runner.py frugal --reward frugal --run_type pid
python micro_runner.py frugal --reward frugal --run_type pid --noise 0.01
python micro_runner.py frugal --reward frugal --run_type pid --noise 0.05
python micro_runner.py sleepy --reward sleepy --num_timesteps 1000000
python micro_runner.py sleepy --reward sleepy --run_type load
python micro_runner.py sleepy --reward sleepy --run_type load --noise 0.01
python micro_runner.py sleepy --reward sleepy --run_type load --noise 0.05
python micro_runner.py sleepy --reward sleepy --run_type pid
python micro_runner.py sleepy --reward sleepy --run_type pid --noise 0.01
python micro_runner.py sleepy --reward sleepy --run_type pid --noise 0.05
python micro_runner.py noisy_frugal1 --reward frugal --num_timesteps 2000000 --noise 0.01
python micro_runner.py noisy_frugal5 --reward frugal --num_timesteps 2000000 --noise 0.05
python micro_runner.py noisy_optimal1 --num_timesteps 2000000 --noise 0.01
python micro_runner.py noisy_optimal5 --num_timesteps 2000000 --noise 0.05
python micro_runner.py noisy_sleepy1 --reward sleepy --num_timesteps 2000000 --noise 0.01
python micro_runner.py noisy_sleepy5 --reward sleepy --num_timesteps 2000000 --noise 0.05
