/**
 * class ImageEncode
 * @author dcsslg
 * Description: Encodes and decodes images using a simple shift-register scheme
 */
public class ImageEncode {
    /**
     * transform
     * @param image is the SimpleImage to manipulate
     * @param shiftReg is the shift register to use in the transformation
     * @throws Exception
     */
    static void transform(SimpleImage image, ILFShiftRegister shiftReg) {
        // If no image, do nothing and return
        if (image == null) {
            return;
        }

        // If no shift register, do nothing and return
        if (shiftReg == null) {
            return;
        }

        // Get the height and width of the image
        int iWidth = image.getImgWidth();
        int iHeight = image.getImgHeight();

        // Catch all exceptions
        try {
            // Iterate over every pixel in the image
            for (int i = 0; i < iWidth; i++) {
                for (int j = 0; j < iHeight; j++) {
                    // For each pixel, get the red, green, and blue components of the color
                    int red = image.getRed(j, i);
                    int green = image.getGreen(j, i);
                    int blue = image.getBlue(j, i);

                    // For each color component, XOR the value with 8 bits generated from the shift register
                    red = (red ^ shiftReg.generate(8));
                    green = (green ^ shiftReg.generate(8));
                    blue = (blue ^ shiftReg.generate(8));

                    // Update the image
                    image.setRGB(j, i, red, green, blue);
                }
            }
        }
        catch(Exception e) {
            // Print out any errors
            System.out.println("Error with transformation: " + e);
        }
    }


    /**
     * main procedure
     * @param args
     */
    public static void main(String[] args) {
        // Open an image
        SimpleImage image = new SimpleImage("Mystery Image", "mystery.bmp");

        // Transform the image using a shift register
        try {
            /*
             * TODO: Add your code here to create a shift register.
             * Use the variable name `shiftReg' for your shift register.
             * Use your shift register implementation, and set
             * the tap and the correct seed.
             */
            ILFShiftRegister shiftReg = new ShiftRegister(13, 7);
            shiftReg.setSeed(new int[] {1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0});
            ////////////////////////////////
            // Transform the image
            transform(image, shiftReg);
        } catch(Exception e) {
            System.out.println("Error in transforming image: " + e);
        }
    }
}
