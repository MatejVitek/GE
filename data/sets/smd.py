from . import Dataset, Sample


class SMDSample(Sample, regex=r'.*/(?P<id>\d+)/[^/]*', regex_on_full_path=True):
	pass


class SMD(Dataset, sample_cls=SMDSample):
	pass
