from . import Dataset, Sample


class SLDSample(Sample, regex=r'(?P<id>\d+)\s*\((?P<n>\d+)\).*'):
	def _postprocess_attributes(self):
		super()._postprocess_attributes()
		self.n = int(self.n)


class SLD(Dataset, sample_cls=SLDSample):
	pass
