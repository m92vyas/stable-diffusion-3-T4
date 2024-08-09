from setuptools import setup,find_packages

setup(
    name               = 'stable-diffusion-3-t4'
    , version          = '1'
    , license          = 'MIT License'
    , author           = "Maharishi Vyas"
    , author_email     = 'maharishi92vyas@gmail.com'
    , packages         = find_packages('src')
    , package_dir      = {'': 'src'}
    , url              = 'https://github.com/m92vyas/stable-diffusion-3-T4.git'
    , keywords         = 'stable diffusion 3 on colab t4 gpu'
    , install_requires = [
                            'diffusers==0.30.0',
                            'torch',
                            'transformers==4.42.4',
                            'lark',
                         ]
    , include_package_data=True
)
