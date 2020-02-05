import numpy as np
import pandas as pd
import folium
import os
import time
import selenium.webdriver
from filepaths import paths

paths = paths(1)

clean = paths.data.clean.path + '/clean.pkl'

df = pd.read_pickle(clean)

states = df['state'].value_counts()
states.sort_index(inplace=True)


states = pd.DataFrame(states)
states.reset_index(inplace=True)

states['state_upper'] = states['index'].str.upper()

states = pd.DataFrame(states.groupby('state_upper').sum())

states = pd.DataFrame(states)
states.reset_index(inplace=True)


url = 'https://raw.githubusercontent.com/'\
      'python-visualization/folium/master/examples/data'
state_geo = f'{url}/us-states.json'

def make_maps():
        m = folium.Map(location=[44, -115],
                       tiles='cartodbpositron',
                       zoom_start=3.6)
        folium.Choropleth(
            geo_data=state_geo,
            name='choropleth',
            data=states,
            columns=['state_upper', 'state'],
            key_on='feature.id',
            fill_color='GnBu',
            fill_opacity=0.7,
            line_opacity=0.2,
            nan_fill_opacity=0.0,
            bins=[x for x in range(0, 350, 50)],
            legend_name='Number of Accidents'
        ).add_to(m)

        folium.LayerControl().add_to(m)

        m.save(paths.images.path + '/map.html')


def make_images():
    list_years = []
    for root, dirs, file in os.walk('./maphtmls'):
        list_years.extend(file)
    list_years.sort()

    delay = 2
    for year in list_years:
        tmpurl = f'file://{srcpath}/visualizations/'\
                 f'maps/maphtmls/CustomerState/{year}'
        print(tmpurl)
        browser = selenium.webdriver.Safari()
        browser.set_window_size(1200, 800)
        browser.get(tmpurl)
        time.sleep(delay)
        browser.save_screenshot(f'{imagepath}/maps/CustomerState/'
                                f'{str(year[:-5])}.png')
        browser.quit()


if __name__ == '__main__':

    make_maps()
    # make_images()
