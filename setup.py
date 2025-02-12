from setuptools import setup, find_packages

setup(
    name='watcher',                        # パッケージの名前
    version='0.1',                            # バージョン
    description='Your package description',   # パッケージの説明
    #author='Your Name',                       # 作者名
    #author_email='your_email@example.com',    # 作者のメールアドレス
    #url='https://github.com/yourusername/my_package',  # リポジトリのURL
    install_requires=[                        # 依存パッケージ
        'numpy',  # 必要に応じて
    ],
    classifiers=[                               # パッケージ情報
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    long_description=open('README.md').read(),  # README.mdの内容を取得
    long_description_content_type='text/markdown',  # READMEの形式を指定
    python_requires='>=3.6',                   # Pythonのバージョン制限


    packages=find_packages(where='src'), 
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'watcher = cli:main',
        ],
    },
)
