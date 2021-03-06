# modification of config created here: https://gist.github.com/cscorley/9144544
try:
    from urllib.parse import quote  # Py 3
except ImportError:
    from urllib2 import quote  # Py 2
import os
import sys

f = None
for arg in sys.argv:
    if arg.endswith('.ipynb'):
        f = arg.split('.ipynb')[0]
        break


c = get_config()
c.NbConvertApp.export_format = 'markdown'
#c.MarkdownExporter.template_path = ['/home/tdos/.ipython/templates'] # point this to your jekyll template file
c.MarkdownExporter.template_path = ['/Users/suhyun/anaconda3/lib/python3.6/site-packages/nbconvert/templates'] # point this to your jekyll template file

c.MarkdownExporter.template_file = 'jekyll'
#c.Application.verbose_crash=True

# modify this function to point your images to a custom path
# by default this saves all images to a directory 'images' in the root of the blog directory
def path2support(path):
    """Turn a file path into a URL"""
    return '{{ BASE_PATH }}/images/' + os.path.basename(path)

c.MarkdownExporter.filters = {'path2support': path2support}

if f:
    c.NbConvertApp.output_base = f.lower().replace(' ', '-')
    #c.FilesWriter.build_directory = '/home/tdos/git/tgarc.github.io/notebooks' # point this to your build directory
    c.FilesWriter.build_directory = '/Users/suhyun/Documents/sh-workspace/github/suhyun-cho.github.io/_posts/kaggle'  # point this to your build directory


