#
# MUSIC𝄞NTWRK
#
# A python library for pitch class set and rhythmic sequences classification and manipulation,
# the generation of networks in generalized music and sound spaces, and the sonification of arbitrary data
#
# Copyright (C) 2018 Marco Buongiorno Nardelli
# http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
def fetchWaves(url):
    # fetch wave files from remote web repository
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page)
    for link in soup.findAll('a'):
        if 'wav' in link.get('href'):
            print(link.get('href'))
            wget.download(link.get('href'))

