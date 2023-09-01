from pathlib import Path
from setuptools import setup

NAME = 'speechtokenizer'
DESCRIPTION = 'Unified speech tokenizer for speech language model'
URL = 'https://github.com/ZhangXInFD/SpeechTokenizer'
EMAIL = 'xin_zhang22@m.fudan.edu.cn'
AUTHOR = 'Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, Xipeng Qiu'
REQUIRES_PYTHON = '>=3.8.0'


for line in open('speechtokenizer/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']
        
HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION
    
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['speechtokenizer', 'speechtokenizer.quantization', 'speechtokenizer.modules'],
    install_requires=['numpy', 'torch', 'torchaudio', 'einops'],
    include_package_data=True,
    license='Apache License 2.0',
    classifiers=[
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
    ])
