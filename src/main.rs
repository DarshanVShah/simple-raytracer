//Creating a simple ray tracer in Rust

use std::{ // Import necessary modules
    fmt::{Display, Debug}, // For formatting output
    fs::File, // For file operations
    io::{BufWriter, Write}, // For buffered writing
};
use glam::DVec3;
use indicatif::{ProgressIterator}; // For progress bar functionality
use itertools::Itertools; // For Cartesian product functionality

fn main() -> std::io::Result<()> { 
    let mut buffer = BufWriter::new(File::create("sample.ppm")?); // Create a buffered writer to write to a file named "sample.ppm"
    let img = basic_scene(); // Generate a sample image using the `sample_image` function
    write!(buffer, "{}", PPM(&img))?; // Write the image data to the buffer in PPM format using the `PPM` struct
    buffer.flush()?; 
    println!("Successfully generated an image"); 
    Ok(()) // Return Ok to indicate successful completion

}

fn basic_scene() -> Image {
    let aspect_ratio = 16.0 / 9.0; // Define the aspect ratio of the image
    let img_width: usize = 400;
    let img_height = (img_width as f64 / aspect_ratio) as usize; // Calculate the height based on the aspect ratio

    //camera
    let focal_len = 1.;
    let viewport_height = 2.;
    let viewport_width = viewport_height * (img_width as f64 / img_height as f64); // Calculate the viewport width based on the aspect ratio
    let camera_center = DVec3::new(0., 0., 0.); // Define the camera center in 3D space

    let viewport_lr = DVec3::new(viewport_width, 0., 0.);
    let viewport_ud = DVec3::new(0., -viewport_height, 0.);

    //calculate delta vectors
    let pixel_delta_lr = &viewport_lr / (img_width as f64); // Calculate the change in the right direction per pixel
    let pixel_delta_ud = &viewport_ud / (img_height as f64); // Calculate the change in the up direction per pixel

    //calculate the location of the upper left corner of the viewport
    let viewport_upper_left = &camera_center - DVec3::new(0.,0.,focal_len) - (&viewport_lr / 2.) - (&viewport_ud / 2.); // Calculate the upper left corner of the viewport

    let pixel00_loc = viewport_upper_left + 0.5 * (&pixel_delta_lr + &pixel_delta_ud); // Calculate the location of the pixel at row 0, column 0

    Image::new_with_init(img_height, img_width, |row, col| {
        let pixel_center = &pixel00_loc + (row as f64) * &pixel_delta_ud + (col as f64) * &pixel_delta_lr; // Calculate the center of the pixel at the given row and column
        let ray_dir = &pixel_center - &camera_center; // Calculate the direction of the ray from the camera center to the pixel center
        let ray = Ray { origin: camera_center, direction: ray_dir }; // Create a new ray with the camera center as the origin and the calculated direction

        let pixel_color = ray.color() * 255.999; // Calculate the color of the pixel based on the ray's color and scale it to 255
        let r = pixel_color.x; // Extract the red component of the color
        let g = pixel_color.y; // Extract the green component of the color
        let b = pixel_color.z; // Extract the blue component of the color

        Pixel { // Create a new Pixel struct with the calculated RGB values
            r: r as u8, // Convert the red component to u8
            g: g as u8, // Convert the green component to u8
            b: b as u8, // Convert the blue component to u8
        }

        

    })
}

// Function to check if a ray intersects with a sphere
fn hit_sphere(center: &DVec3, radius: f64, ray: &Ray) -> f64 {
    let origin_center = ray.origin - *center; // Calculate the vector from the ray's origin to the sphere's center
    let a = ray.direction.length_squared(); // Calculate the dot product of the ray's direction with itself
    let half_b = origin_center.dot(ray.direction); // Calculate the dot product of the origin-center vector with the ray's direction, multiplied by 2
    let c = origin_center.length_squared() - radius * radius; // Calculate the dot product of the origin-center vector with itself, minus the square of the radius

    let discriminant = half_b * half_b - a * c; // Calculate the discriminant of the quadratic equation
    
    if discriminant < 0. { // If the discriminant is negative, there is no intersection
        -1.0 // Return -1 to indicate no intersection
    } else {
        (- half_b - discriminant.sqrt()) / a // Calculate the intersection point using the quadratic formula
    }
}

// Function to create a sample image
fn sample_image() -> Image { 
    let image_width = 256;
    let image_height = 256;
    Image::new_with_init(image_height, image_width, |row, col| { // Initialize each pixel with a color based on its position
        let r = col as f64 / ((image_width - 1) as f64); 
        let g = row as f64 / ((image_height - 1) as f64); 
        let b = 0.;

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

struct Ray { // Represents a ray in 3D space
    origin: DVec3,
    direction: DVec3,
}

impl Ray { // Create a new Ray with a given origin and direction
    fn at(&self, t: f64) -> DVec3 {
        self.origin + t * self.direction
    }

    fn color(&self) -> DVec3 {
        let t = hit_sphere(&DVec3::new(0., 0., -1.), 0.5, self);
        if t > 0.0 { // If the ray intersects with the sphere
            let normal = (self.at(t) - DVec3::new(0., 0., -1.)).normalize(); // Calculate the normal vector at the intersection point
            return 0.5 * (normal + 1.0); // Return a color based on the normal vector
        };

        let unit_direction: DVec3 =
            self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - a) * DVec3::new(1.0, 1.0, 1.0)
            + a * DVec3::new(0.5, 0.7, 1.0);
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
