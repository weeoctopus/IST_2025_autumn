"""
Скрипт для скачивания данных LIBSVM
"""
import os
import urllib.request
import bz2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def download_libsvm_data(dataset_name, data_dir="./data"):
    """
    Скачивание данных из LIBSVM.
    """
    urls = {
        'w8a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a',
        'gisette': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2',
        'real-sim': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2'
    }
    
    # Создаем директорию если нужно
    os.makedirs(data_dir, exist_ok=True)
    
    url = urls.get(dataset_name)
    if not url:
        print(f"Unknown dataset: {dataset_name}")
        return False
    
    filename = url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    
    print(f"Downloading {dataset_name} from {url}...")
    
    try:
        # Скачиваем файл
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")
        
        # Распаковываем если это .bz2
        if filename.endswith('.bz2'):
            print(f"Extracting {filepath}...")
            with bz2.open(filepath, 'rb') as f:
                decompressed_data = f.read()
            
            # Убираем .bz2 из имени файла
            output_path = filepath[:-4]
            with open(output_path, 'wb') as f:
                f.write(decompressed_data)
            
            print(f"Extracted to {output_path}")
            # Удаляем архив
            os.remove(filepath)
        
        return True
        
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")
        return False

def download_all_datasets():
    """Скачивание всех датасетов."""
    datasets = ['w8a', 'gisette', 'real-sim']
    
    print("Downloading LIBSVM datasets...")
    print("="*50)
    
    for dataset in datasets:
        success = download_libsvm_data(dataset)
        if success:
            print(f"✓ {dataset} downloaded successfully")
        else:
            print(f"✗ Failed to download {dataset}")
        print()

if __name__ == "__main__":
    download_all_datasets()
    print("\nPlease run the test script after downloading the data.")