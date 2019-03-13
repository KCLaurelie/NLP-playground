import sys
from PyPDF2 import PdfFileMerger,PdfFileReader,PdfFileWriter
Path='D:\\Biological Foundations of Mental Health\\'
Pdfs=['M1-W2-T1-Brenda-Williams-Part1of4-Transcript.pdf','M1R1-W2-T1-Parts1to4-Sandrine-Thuret-and-Brenda-Williams.pdf']
Out='out.pdf'

Merger=PdfFileMerger()
for i in Pdfs:
  Merger.append(Path+i)
Merger.write(Path+Out)
