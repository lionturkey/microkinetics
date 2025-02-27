








if run_name is not None:
    self.run_folder = Path.cwd() / 'runs' / run_name
    self.run_folder.mkdir(parents=True, exist_ok=True)