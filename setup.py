from setuptools import find_packages, setup

def get_version(path):
    for line in open(path):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError(f"Unable to find __version__ in {path}")


def get_requirements(path):
    res = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('--'):
            continue
        res.append(line.split('#')[0].strip())
    return res


setup(
    name='evoagentx',
    version=get_version('evoagentx/__init__.py'),
    author='Jinyuan Fang',
    author_email='fangjy6@gmail.com',
    description="EvoAgentX: An Evolutionary Agent Framework",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/EvoAgentX/EvoAgentX',
    packages=find_packages(),
    include_package_data=True,
    entry_points={},
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.10',
)