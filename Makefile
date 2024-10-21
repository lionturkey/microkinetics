task:
	date

setupenv:
	conda create -n micro -c conda-forge --file requirements.txt
	conda activate micro
	