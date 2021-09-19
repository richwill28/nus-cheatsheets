import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import javax.swing.*;
import java.io.File;
import java.io.IOException;

/**
 * Class SimpleImage
 * @author dcsslg
 * Description: A simple class for loading an image from disk and getting/setting the pixels
 *
 */
public class SimpleImage extends Component {
    // Unique id used for serialization
    private static final long serialVersionUID = 1L;

    // Stores the encapsulated image
    BufferedImage img = null;

    /**
     * Constructor: creates a new SimpleImage
     * @param windowName is the title of the image window.
     * @param fileName is the name of the file to open.
     */
    SimpleImage(String windowName, String fileName) {
        // Open the file
        openFile(fileName);

        // Create a new frame with the specified name
        JFrame f = new JFrame(windowName);

        // Setup this window
        f.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });

        // Tell the window what to display
        f.add(this);

        // Activate the window
        f.pack();
        f.setVisible(true);
    }

    /**
     * openFile: opens the specified file
     * @param fileName is the name of the file to open
     */
    private void openFile(String fileName) {
        // Initialize the image to null (in case a previous image was opened).
        img = null;

        // Catch any exceptions.
        try {
            // Read the file
            img = javax.imageio.ImageIO.read(new File(fileName));
        } catch (IOException e) {
            // If there are any exceptions, do nothing.
            // The image will remain empty.
        }
    }

    /**
     * saveFile: saves the file with the specified name
     * @param fileName is the name of the file to save
     */
    public void saveFile(String fileName) {
        try {
            javax.imageio.ImageIO.write(img, "bmp", new File(fileName));
        } catch (IOException e) {
            // If there are any exceptions, do nothing.
        }
    }

    /**
     * getPreferredSize returns the dimension of the window.
     */
    public Dimension getPreferredSize() {
        if (img == null) {
            return new Dimension(100, 100);
        } else {
            return new Dimension(img.getWidth(null), img.getHeight(null));
        }
    }

    /**
     * paint draws the image onto the specified graphics context.
     */
    public void paint(Graphics g) {
        g.drawImage(img, 0, 0, null);
    }

    /**
     * getRed
     * @param row is the row in the image.
     * @param column is the column in the image.
     * @return the value of the red component of the image at the specified location
     */
    public int getRed(int row, int column) {
        if (img == null) {
            return 0;
        }

        //Get the color from the image
        Color c = new Color(img.getRGB(column, row));
        return c.getRed();
    }

    /**
     * getBlue
     * @param row is the row in the image.
     * @param column is the column in the image.
     * @return the value of the blue component of the image at the specified location
     */
    public int getBlue(int row, int column) {
        if (img == null) {
            return 0;
        }

        //Get the color from the image
        Color c = new Color(img.getRGB(column, row));
        return c.getBlue();
    }

    /**
     * getGreen
     * @param row is the row in the image.
     * @param column is the column in the image.
     * @return the value of the red component of the image at the specified location
     */
    public int getGreen(int row, int column) {
        if (img == null) {
            return 0;
        }

        // Get the color from the image
        Color c = new Color(img.getRGB(column, row));
        return c.getGreen();
    }

    /**
     * setRed
     * @param row is the row in the image
     * @param column is the column in the image
     * @param v is the value of the red component to set
     */
    public void setRed(int row, int column, int v) {
        if (img == null) {
            return;
        }

        // Maximum value: 255
        if (v > 255) {
            v = 255;
        }

        // Minimum value: 0
        if (v < 0) {
            v = 0;
        }

        // Get the old color
        Color c = new Color(img.getRGB(column, row));

        // Create an integer representation of the new color.  Each of the colors
        // is stored in the following bits.
        // Blue: [7:0]
        // Green: [15:8]
        // Red: [23:16]
        int newColor = c.getBlue() + (c.getGreen() * 256) + (v * 256 * 256);
        img.setRGB(column, row, newColor);

        // Redraw the image in the window
        repaint();
    }

    /**
     * setBlue
     * @param row is the row in the image
     * @param column is the column in the image
     * @param v is the value of the blue component to set
     */
    public void setBlue(int row, int column, int v) {
        if (img == null){
            return;
        }

        // Maximum value: 255
        if (v > 255) {
            v = 255;
        }

        // Minimum value: 0
        if (v < 0) {
            v = 0;
        }

        // Get the old color
        Color c = new Color(img.getRGB(column, row));

        // Create an integer representation of the new color.  Each of the colors
        // is stored in the following bits.
        // Blue: [7:0]
        // Green: [15:8]
        // Red: [23:16]
        int newColor = v + (c.getGreen() * 256) + (c.getRed() * 256 * 256);
        img.setRGB(column, row, newColor);

        // Redraw the image in the window
        repaint();
    }

    /**
     * setGreen
     * @param row is the row in the image
     * @param column is the column in the image
     * @param v is the value of the green component to set
     */
    public void setGreen(int row, int column, int v) {
        if (img == null) {
            return;
        }

        // Maximum value: 255
        if (v > 255) {
            v = 255;
        }

        // Minimum value: 0
        if (v < 0) {
            v = 0;
        }

        // Get the old color
        Color c = new Color(img.getRGB(column, row));

        // Create an integer representation of the new color.  Each of the colors
        // is stored in the following bits.
        // Blue: [7:0]
        // Green: [15:8]
        // Red: [23:16]
        int newColor = c.getBlue() + (v * 256) + (c.getRed() * 256 * 256);
        img.setRGB(column, row, newColor);

        // Redraw the image in the window
        repaint();
    }

    /**
     * setRGB sets all three colors at once
     * @param row is the row in the image
     * @param column is the column in the image
     * @param red is the 8-bit red value
     * @param green is the 8-bit green value
     * @param blue is the 8-bit blue value
     */
    public void setRGB(int row, int column, int red, int green, int blue) {
        if (img == null) {
            return;
        }

        // Maximum value: 255
        if (red > 255) {
            red = 255;
        }
        if (green > 255) {
            green = 255;
        }
        if (blue > 255) {
            blue = 255;
        }

        // Minimum value: 0
        if (red < 0) {
            red = 0;
        }
        if (green < 0) {
            green = 0;
        }
        if (blue < 0) {
            blue = 0;
        }

        // Create the color represented as an integer
        int newColor = blue + (green * 256) + (red * 256 * 256);

        // Set the color
        img.setRGB(column, row, newColor);

        // Redraw the window
        repaint();
    }

    /**
     * getImgWidth
     * @return the width of the image
     */
    public int getImgWidth() {
        if (img == null) {
            return 0;
        }
        return img.getWidth();
    }

    /**
     * getImgHeight
     * @return the height of the image
     */
    public int getImgHeight() {
        if (img == null) {
            return 0;
        }
        return img.getHeight();
    }
}
