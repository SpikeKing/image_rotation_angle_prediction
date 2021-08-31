#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 23.7.21
"""

from myutils.project_utils import read_file


def make_html_page(html_file, img_data_list, n=1):
    header = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>MathJax TeX Test Page</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script>
    <style>
    img{
        max-height:1200px;
        max-width: 1200px;
        vertical-align:middle;
    }
    </style>
    </head>
    <body>
    <table border="1">
    """

    tail = """
    </table>
    </body>
    </html>
    """

    # data_lines = read_file(img_data_list)
    print('[Info] 样本行数: {}'.format(len(img_data_list)))
    urls_list = img_data_list  # url列表

    with open(html_file, 'w') as f:
        f.write(header)
        for idx, items in enumerate(urls_list):
            f.write('<tr>\n')
            f.write('<td>%d</td>\n' % ((idx / n) + 1))
            for item in items:
                item = str(item)
                f.write('<td>\n')
                if item.startswith("http"):
                    f.write('<img src="%s" width="600">\n' % item)
                else:
                    f.write('%s' % item)
                f.write('</td>\n')
            f.write('</tr>\n')
        f.write(tail)