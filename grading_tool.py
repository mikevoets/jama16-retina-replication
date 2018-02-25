import PIL.Image
from tkinter import *
from tkinter import filedialog
import PIL.ImageTk
import os
import sys
from glob import glob
import csv
from shutil import copyfile

class GradingTool(Frame):
    def chg_image(self):
        self.image = PIL.ImageTk.PhotoImage(self.im)
        self.label.config(image=self.image, bg="#000000",
                          width=self.image.width(), height=self.image.height())

    def open_image(self, i):
        filename = self.im_paths[i]
        self.im = PIL.Image.open(filename)
        self.page_num = i
        self.chg_image()
        self.page_num_strf.set("{}/{}".format(self.page_num + 1, self.im_count))
        self.image_name_strf.set(os.path.basename(filename))

    def open(self):
        d = filedialog.askdirectory(initialdir=os.getcwd(),
                                    title='Select folder', mustexist=1)
        if d:
            self.dataset_name = "_".join(d.split("/")[-2:])
            self.im_paths = sorted(
                                glob(os.path.join(d, "**/*.jp*g"),
                                     recursive=True))
            self.im_count = len(self.im_paths)
            self.gradable_dir = "./.gt/{}_gradable".format(self.dataset_name)
            self.csv_filename = "./.gt/{}.csv".format(self.dataset_name)
            self.checkp_filename = "./.gt/{}_checkp.txt".format(self.dataset_name)

            if os.path.exists(self.gradable_dir):
                checkpoint = self.get_checkpoint()
                self.page_num = checkpoint + 1

                if self.page_num == self.im_count:
                    print("Already graded entire data set!")
                    sys.exit(2)

                print("Continue at {}".format(self.page_num + 1))
            else:
                os.makedirs(self.gradable_dir)
                self.page_num = 0

            self.csvfile = open(self.csv_filename, 'a')
            self.csv = csv.writer(self.csvfile, delimiter=' ')

            self.open_image(self.page_num)

    def copy_images(self):
        with open(self.csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            for row in reader:
                im_path, gradable = row[:2]
                im_path = os.path.relpath(im_path)

                if gradable == "1":
                    directory = "/".join(im_path.split("/")[:-1])

                    if not os.path.exists(os.path.join(
                                self.gradable_dir, directory)):
                        os.makedirs(os.path.join(self.gradable_dir, directory))

                    copyfile(im_path, os.path.join(self.gradable_dir, im_path))

    def continue_later(self):
        self.csvfile.close()
        print("Progress saved. Continue grading later.")
        sys.exit(0)

    def finalize(self):
        self.csvfile.close()
        self.copy_images()
        print("Done grading! Gradable images are copied to {}"
              .format(self.gradable_dir))
        sys.exit(0)

    def write_checkpoint(self, i):
        with open(self.checkp_filename, 'w') as f:
            f.write(str(i))

    def get_checkpoint(self):
        with open(self.checkp_filename, 'r') as f:
            checkpoint = f.read()
        return int(checkpoint)

    def get_next(self):
        self.write_checkpoint(self.page_num)
        try:
            self.open_image(self.page_num + 1)
        except IndexError:
            self.finalize()

    def gradable(self):
        self.csv.writerow([self.im_paths[self.page_num], '1'])
        self.get_next()

    def not_gradable(self):
        self.csv.writerow([self.im_paths[self.page_num], '0'])
        self.get_next()

    def __init__(self):
        Frame.__init__(self)
        self.master.title('Grading Tool')

        self.page_num = 0
        self.page_num_strf = StringVar()
        self.image_name_strf = StringVar()

        frame = Frame(self)
        Button(frame, text="Open Folder", command=self.open).pack(side=LEFT)
        Button(frame, text="Gradable", command=self.gradable).pack(side=LEFT)
        Button(frame, text="Not Gradable", command=self.not_gradable).pack(side=LEFT)
        frame.pack(side=TOP, fill=BOTH)

        frame2 = Frame(self)
        Label(frame2, textvariable=self.image_name_strf).pack(side=LEFT)
        Label(frame2, textvariable=self.page_num_strf).pack(side=RIGHT)
        frame2.pack(side=TOP, fill=BOTH)

        self.label = Label(self)
        self.label.pack()

        frame3 = Frame(self)
        Button(frame3, text="Continue Later", command=self.continue_later).pack(side=LEFT)
        frame3.pack(side=TOP, fill=BOTH)

        self.pack()


if __name__ == "__main__":
    tool = GradingTool(); tool.mainloop()
