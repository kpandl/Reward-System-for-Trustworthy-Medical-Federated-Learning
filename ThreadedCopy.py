import queue, threading, os, time
import shutil


class ThreadedCopy:
    totalFiles = 0
    copyCount = 0
    lock = threading.Lock()
    fileQueue = queue.Queue()

    def __init__(self, file_list, destPath):

        if not os.path.exists(destPath):
            os.mkdir(destPath)

        self.destPath = destPath

        self.totalFiles = len(file_list)

        print(str(self.totalFiles) + " files to copy.")
        self.threadWorkerCopy(file_list)

    def CopyWorker(self):
        while True:
            fileName = self.fileQueue.get()
            if(os.path.exists(fileName)):
                shutil.copy(fileName, self.destPath)
            #print("copied", fileName, "to", self.destPath)
            self.fileQueue.task_done()
            with self.lock:
                self.copyCount += 1
                #percent = (self.copyCount * 100) / self.totalFiles
                #print(str(percent) + " percent copied.")

    def threadWorkerCopy(self, fileNameList):
        for i in range(32):
            t = threading.Thread(target=self.CopyWorker)
            t.daemon = True
            t.start()
        for fileName in fileNameList:
            self.fileQueue.put(fileName)
        self.fileQueue.join()