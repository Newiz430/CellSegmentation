// QuPath script for extracting annotations.
setImageType('BRIGHTFIELD_H_DAB');
setColorDeconvolutionStains('{"Name" : "H-DAB default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049 ", "Stain 2" : "DAB", "Values 2" : "0.26917 0.56824 0.77759 ", "Background" : " 255 255 255 "}');
def result1 = ''
def result2 = ''
annotations = getAnnotationObjects()
for (annotation in getAnnotationObjects()) {
    def pathClass = annotation.getPathClass()
    def roi = annotation.getROI()
    //result1 += roi.getPolygonPoints()
    result1 += roi.getAllPoints() // Use when all annotations are polygons
    result1 += System.lineSeparator()
//    result2 += String.format('%s, %s, %.2f, %.2f, %.2f, %.2f',
//        pathClass, annotation, roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight())
//    result2 += System.lineSeparator()

}
def imageData = getCurrentImageData()
//def server = imageData.getServer()
//def splitname = server.path.split('\\\\')
//def name = splitname[splitname.length-2] + "_" + splitname[splitname.length-1].split('\\.')[0]
def name = imageData.getServer().getMetadata().name.split('\\.')[0]
def file1 = new File("E://Code//python//CellSegmentation//annotate//" + name + ".csv")
file1.text = result1
//def file2 = new File("E://Code//python//CellSegmentation//annotate//" + name + "2.csv")
//file2.text = result2