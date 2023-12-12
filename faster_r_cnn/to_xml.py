import os
import xml.etree.ElementTree as ET

def convert_txt_to_xml(txt_folder, img_folder, xml_folder):
    # List all .txt files in the txt_folder
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        # Extract base filename without extension
        base_filename = os.path.splitext(txt_file)[0]
        
        # Open the .txt file and read lines
        with open(os.path.join(txt_folder, txt_file)) as file:
            lines = file.readlines()
        
        # Create the root of the XML
        annotation = ET.Element('annotation')
        
        # Iterate over each line in the txt file
        for line in lines:
            class_id, x_center, y_center, width, height = line.split()
            
            # Convert from yolo format to Pascal VOC format
            # Assuming image width and height are known, need to be provided
            image_width, image_height = 1024, 1024  # Replace with actual image dimensions
            
            x_min = float(x_center) - (float(width) / 2)
            y_min = float(y_center) - (float(height) / 2)
            x_max = float(x_center) + (float(width) / 2)
            y_max = float(y_center) + (float(height) / 2)
            
            # Scale to original image size
            x_min *= image_width
            y_min *= image_height
            x_max *= image_width
            y_max *= image_height

            # Create 'object' element
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = str(class_id)
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            # Create 'bndbox' element
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(x_min))
            ET.SubElement(bndbox, 'ymin').text = str(int(y_min))
            ET.SubElement(bndbox, 'xmax').text = str(int(x_max))
            ET.SubElement(bndbox, 'ymax').text = str(int(y_max))
        
        # Create a new XML file for each annotation
        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(xml_folder, base_filename + '.xml'))
    
    return f"Converted {len(txt_files)} TXT files to XML."

# Define the folders (replace with the actual paths)
txt_folder_path = 'yolo_person_train\\anno\labels'
img_folder_path = 'yolo_person_train\\anno\images'
xml_folder_path = 'yolo_person_train\\annotations'

# Call the conversion function
convert_txt_to_xml(txt_folder_path, img_folder_path, xml_folder_path)
