from tkinter import * 
import customtkinter as ctk
import os



#folderpath = ctk.filedialog.askdirectory()






def choose_folder():
    folder = ctk.filedialog.askdirectory()
    if folder:
        folder_path.set(folder)
        update_file_list(folder)

def update_file_list(folder):
    file_list['values'] = os.listdir(folder)



def on_click_runButton():
    #! THIS IS WHERE I CALL BINARIZE SKELETONIZE AND FEATURE EXTRACT. etc
    print('run button was clicked successfully!')



root = ctk.CTk()
root.title("Astrocyte Software")


folder_path = ctk.StringVar()

frame = ctk.CTkFrame(root, width=200, height=200)
frame.grid(row=10,column=0, sticky=(ctk.W, ctk.E, ctk.N,ctk.S))


ctk.CTkLabel(frame, text="Directory..").grid(row=0,column=0, sticky=ctk.W)
ctk.CTkEntry(frame, textvariable=folder_path, width=30).grid(row=0, column=1,sticky=(ctk.W,ctk.E))

ctk.CTkButton(frame, text="Choose...", command=choose_folder).grid(row=0, column=2, sticky=ctk.E)


ctk.CTkLabel(frame, text="Files:").grid(row=1, column=0, sticky=ctk.W)
file_list = ctk.CTkComboBox(frame, values=[])
file_list.grid(row=1, column=1, sticky=(ctk.W, ctk.E))
runButton = ctk.CTkButton(root, text="Click me to run!", command=on_click_runButton)

root.mainloop()








root.mainloop()
