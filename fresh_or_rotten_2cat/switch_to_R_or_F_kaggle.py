import os

def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Change the file extension if required
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r+') as file:
                first_char = file.read(1)
                if first_char.isdigit():
                    first_digit = int(first_char)
                    if 0 <= first_digit <= 2:
                        file.seek(0)
                        file.write('0')
                    elif 3 <= first_digit <= 5:
                        file.seek(0)
                        file.write('1')

# Replace '/path/to/folder' with the actual folder path
process_files_in_folder('./test/labels')
process_files_in_folder('./train/labels')
