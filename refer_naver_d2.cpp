/*
See : http://d2.naver.com/helloworld/8344782
*/
class PreProcess {

    private void testContour(Mat imageMat) {
        Mat rgb = new Mat();  //rgb color matrix
        rgb = imageMat.clone();
        Mat grayImage = new Mat();  //grey color matrix
        Imgproc.cvtColor(rgb, grayImage, Imgproc.COLOR_RGB2GRAY);

        Mat gradThresh = new Mat();  //matrix for threshold
        Mat hierarchy = new Mat();    //matrix for contour hierachy
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        //Imgproc.threshold(grayImage,gradThresh, 127,255,0);  global threshold
        Imgproc.adaptiveThreshold(grayImage, gradThresh, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 3, 12);  //block size 3
        Imgproc.findContours(gradThresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
        if(contours.size()>0) {
            for(int idx = 0; idx < contours.size(); idx++) {
                Rect rect = Imgproc.boundingRect(contours.get(idx));
                if (rect.height > 10 && rect.width > 40 && !(rect.width >= 512 - 5 && rect.height >= 512 - 5)){
                    rectangle(imageMat, new Point(rect.br().x - rect.width, rect.br().y - rect.height)
                            , rect.br()
                            , new Scalar(0, 255, 0), 5);
                }

            }
            Imgcodecs.imwrite("/tmp/dev/doc_original.jpg", rgb);
            Imgcodecs.imwrite("/tmp/dev/doc_gray.jpg", grayImage);
            Imgcodecs.imwrite("/tmp/dev/doc_thresh.jpg", gradThresh);
            Imgcodecs.imwrite("/tmp/dev/doc_contour.jpg", imageMat);
        }
    }// end of method


    public void removeVerticalLines(Mat img, int limit) {
        Mat lines=new Mat();
        int threshold = 100; //선 추출 정확도
        int minLength = 80; //추출할 선의 길이
        int lineGap = 5; //5픽셀 이내로 겹치는 선은 제외
        int rho = 1;
        Imgproc.HoughLinesP(img, lines, rho, Math.PI/180, threshold, minLength, lineGap);
        for (int i = 0; i < lines.total(); i++) {
            double[] vec=lines.get(i,0);
            Point pt1, pt2;
            pt1=new Point(vec[0],vec[1]);
            pt2=new Point(vec[2],vec[3]);
            double gapY = Math.abs(vec[3]-vec[1]);
            double gapX = Math.abs(vec[2]-vec[0]);
            if(gapY>limit && limit>0) {
                //remove line with black color
                Imgproc.line(img, pt1, pt2, new Scalar(0, 0, 0), 10);
            }
        }
    }// end of method

}// end of class

