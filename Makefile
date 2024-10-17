task:
	date

setupenv:
	conda create -n micro --file requirements.txt
	conda activate micro
	