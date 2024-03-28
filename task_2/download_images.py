import aiohttp
import asyncio
import os
import pandas as pd
import re
from tqdm import tqdm

def clean_filename(filename):
    filename = str(filename)
    return re.sub(r'[<>:"/\\|?*]', '', filename)

async def download_image(session, row, image_dir, max_length=250, retries=3, backoff_factor=0.3):
    image_url = row['iiifthumburl']
    title = clean_filename(row['title'])
    artist = clean_filename(row['attribution'])
    image_name = f"{title}_{artist}.jpg"

    if len(image_name) > max_length:
        image_name = image_name[:max_length] + '.jpg'

    file_path = os.path.join(image_dir, image_name)

    for attempt in range(retries):
        try:
            if not os.path.exists(file_path):
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=60)) as response:  
                    if response.status == 200:
                        content = await response.read()
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        return 1  
                    else:
                        print(f"Failed to download {image_name}: Status code {response.status}")
            return 0  
        except asyncio.TimeoutError:
            if attempt < retries - 1:
                await asyncio.sleep(backoff_factor * (2 ** attempt))  
            else:
                print(f"Failed to download {image_name} after {retries} attempts due to timeout.")
                return 0 

async def main(merged_df, image_dir):
    connector = aiohttp.TCPConnector(limit_per_host=20) 
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image(session, row, image_dir) for _, row in merged_df.iterrows()]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f

published_images_df = pd.read_csv('data/published_images.csv')
objects_df = pd.read_csv('data/objects.csv')

merged_df = pd.merge(published_images_df, objects_df, left_on='depictstmsobjectid', right_on='objectid', how='inner')

image_dir = 'downloaded_images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

asyncio.run(main(merged_df, image_dir))

print("Download completed.")
