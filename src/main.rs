//Creating a simple ray tracer in Rust

use std::{ // Import necessary modules
    fmt::{Display, Debug}, // For formatting output
    fs::File, // For file operations
    io::{BufWriter, Write}, // For buffered writing
};
use indicatif::{ProgressIterator}; // For progress bar functionality
use itertools::Itertools; // For Cartesian product functionality
fn main() -> std::io::Result<()> { 
    let mut buffer = BufWriter::new(File::create("sample.ppm")?); // Create a buffered writer to write to a file named "sample.ppm"
    let img = sample_image(); // Generate a sample image using the `sample_image` function
    write!(buffer, "{}", PPM(&img))?; // Write the image data to the buffer in PPM format using the `PPM` struct
    buffer.flush()?; 
    println!("Successfully generated an image"); 
    Ok(()) // Return Ok to indicate successful completion

}

// Function to create a sample image
fn sample_image() -> Image { 
    let image_width = 256;
    let image_height = 256;
    Image::new_with_init(256, 256, |row, col| { // Initialize each pixel with a color based on its position
        let r = col as f64 / ((image_width - 1) as f64); 
        let g = row as f64 / ((image_height - 1) as f64); 
        let b = 0.9;

        let factor = 255.999; // Scale factor for RGB values

        let r = (r * factor) as u8; 
        let g = (g * factor) as u8;
        let b = (b * factor) as u8;
        Pixel { r, g, b, } // Create a Pixel with the calculated RGB values
    })
}

#[derive(Default)] // Default implementation for Pixel struct

struct Pixel { // Represents a pixel in the image
    // Each pixel has three components: red, green, and blue. In byte values.
    r: u8, 
    g: u8,
    b: u8,
}

struct Image { // Represents an image made up of pixels
    pixels: Vec<Vec<Pixel>>, //2D vector of pixels
}

// Implementing methods for the Image struct
impl Image { 
    pub fn new(height: usize, width: usize) -> Self { // Create a new image with specified height and width
        assert!( 
            height > 0 && width > 0,
            "Height and width must be greater than zero"
        );

        let mut pixels = Vec::with_capacity(height); // Initialize a vector to hold the rows of pixels
       
        // Fill the vector with rows of default pixels
        for _ in 0..height { 
            let mut row = Vec::with_capacity(width); 
            for _ in 0..width { 
                row.push(Pixel::default()) 
            }
            pixels.push(row); 
        }

        Self { pixels } // Return the new Image instance with the initialized pixels
    }

    // Create a new image with specified height and width, initializing each pixel using the provided function
    pub fn new_with_init(
        height: usize,
        width: usize,
        init: impl Fn(usize, usize) -> Pixel,
    ) -> Self {
        let mut image = Self::new(height, width); 

        (0..height)
            .cartesian_product(0..width) // Generate all combinations of row and column indices
            .progress_count((height * width) as u64) // Use a progress bar to track the initialization process
            .for_each(|(row, col)| {
                image.pixels[row][col] = init(row, col);
            });
    
        image
    }

    
    pub fn height(&self) -> usize { // Return the height of the image
        self.pixels.len() 
    }

    pub fn width(&self) -> usize { // Return the width of the image
        self.pixels[0].len()
    }
}

// Implementing Display and Debug traits for PPM (Portable Pixmap) format
struct PPM<'a, T>(&'a T); 
impl Display for PPM<'_, Pixel> { 
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { 
        write!(f, "{:>3} {:>3} {:>3}", self.0.r, self.0.g, self.0.b) // Format pixel values with right alignment
    }
}

impl Debug for PPM<'_, Pixel> { 
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { 
        write!(f, "{}", self) 
    }
}

impl Display for PPM<'_, Image> { 
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { 
        writeln!(f, "P3")?; 
        writeln!(f, "{} {}", self.0.width(), self.0.height())?; 
        writeln!(f, "255")?;

        for row in 0..self.0.height() { 
            for col in 0..self.0.width() { 
                writeln!(f, "{}", PPM(&self.0.pixels[row][col]))?; // Use PPM for each pixel
            }
        }

        Ok(())
    }
}

impl Debug for PPM<'_, Image> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

// Unit tests for the Pixel and Image structs, and the PPM formatting
#[cfg(test)]
mod test { 
    use super::*; 

    #[test]
    fn test1() {
        let pixel = Pixel { r: 42, g: 2, b: 3 };
        k9::snapshot!(PPM(&pixel), " 42   2   3");
    }

    #[test]
    fn test2() {
        let img = Image::new_with_init(2, 3, |row, col| Pixel {
            r: row as u8,
            g: col as u8,
            b: 42,
        });

        k9::snapshot!(
            PPM(&img),
            "
P3
3 2
255
  0   0  42
  0   1  42
  0   2  42
  1   0  42
  1   1  42
  1   2  42

"
        );
    }
}
