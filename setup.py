from setuptools import setup, find_packages
 
# from pkg_resources import parse_requirements
# with open("requirements.txt", encoding="utf-8") as fp:
#     install_requires = [str(requirement) for requirement in parse_requirements(fp)]
 
setup(
    name="aortic dissection",
    version="0.1.0",
    author="Yichen Yang",
    author_email="yangych12022@shanghaitech.edu.cn",
    description="Personal research usage",
    long_description="Personal research usage",
 
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "Operating System :: OS Independent",
    # ],
    
    
    # include_package_data=True, # 一般不需要
    packages=find_packages(),
    # install_requires=install_requires,
)