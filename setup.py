import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name="MCML",
	version="0.0.1",
	author="Tara Chari",
	author_email="tarachari3@gmail.com",
	description="Semi-supervised Dimensionality Reduction for Multi-Class, Multi-Label Data",
	long_description=long_description,
	long_description_content_type="text/markdown",
	install_requires=[
		'setuptools>=49',
		'wheel',
		'torch>=1.9.0',
		'numpy>=1.21.0',
		'pandas>=1.2.5',
		'matplotlib>=3.4.2',
		'torchsummary>=1.5.1',
		'scikit-learn>=0.24.2'
	],
	url="https://github.com/pachterlab/MCML",
	project_urls={
		"Bug Tracker": "https://github.com/pachterlab/MCML/issues",
	},
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: BSD License",
		"Operating System :: OS Independent",
	],
	package_dir={"": "src"},
	packages=setuptools.find_packages(where="src"),
	python_requires=">=3.7",
)
