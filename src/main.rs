//Creating a simple ray tracer in Rust

use glam::DVec3;
use indicatif::ProgressIterator; // For progress bar functionality
use itertools::Itertools;
use rand::prelude::*; // For random number generation
use k9::snapshot::source_code::Range;
use std::{
    // Import necessary modules
    fmt::{Debug, Display},  // For formatting output
    fs::File,               // For file operations
    io::{BufWriter, Write}, // For buffered writing
    ops::Range as StdRange, // Rename Range to StdRange to avoid conflict
}; // For Cartesian product functionality

fn main() -> std::io::Result<()> {
    let mut buffer = BufWriter::new(File::create("sample.ppm")?); // Create a buffered writer to write to a file named "sample.ppm"
    let img = basic_scene(); // Generate a sample image using the `sample_image` function
    write!(buffer, "{}", PPM(&img))?; // Write the image data to the buffer in PPM format using the `PPM` struct
    buffer.flush()?;
    println!("Successfully generated an image");
    Ok(()) // Return Ok to indicate successful completion
}

struct Camera {
    img_width: usize,
    img_height: usize,
    max_value: f64,
    aspect_ratio: f64,
    camera_center: DVec3,
    pixel_delta_lr: DVec3,
    pixel_delta_ud: DVec3,
    pixel00_loc: DVec3,
    samples_per_pixel: u32, // Number of samples per pixel for anti-aliasing
}

impl Camera {
    fn new(img_width: usize, aspect_ratio: f64) -> Self {
        // Create a new Camera instance with the specified parameters
        let focal_len = 1.0; // Define the focal length of the camera
        let max_value = 255.999; // Define the maximum value for pixel color components
        let img_height = (img_width as f64 / aspect_ratio) as usize; // Calculate the height of the image based on the aspect ratio
        let viewport_height = 2.0; // Define the height of the viewport
        let viewport_width = viewport_height * aspect_ratio; // Calculate the width of the viewport based on the aspect ratio
        let camera_center = DVec3::new(0.0, 0.0, 0.0); // Define the center of the camera in 3D space

        let viewport_lr = DVec3::new(viewport_width, 0.0, 0.0); // Define the right direction vector of the viewport
        let viewport_ud = DVec3::new(0.0, -viewport_height, 0.0); // Define the up direction vector of the viewport

        // Calculate delta vectors for pixel size
        let pixel_delta_lr = &viewport_lr / (img_width as f64);
        let pixel_delta_ud = &viewport_ud / (img_height as f64);

        // Calculate the location of the upper left corner of the viewport
        let viewport_upper_left =
            &camera_center - DVec3::new(0.0, 0.0, focal_len) - (&viewport_lr / 2.0) - (&viewport_ud / 2.0);

        // Calculate the location of pixel (0, 0)
        let pixel00_loc =
            viewport_upper_left + 0.5 * (&pixel_delta_lr + &pixel_delta_ud);

        Self {
            img_width,
            img_height,
            max_value,
            aspect_ratio,
            camera_center,
            pixel_delta_lr,
            pixel_delta_ud,
            pixel00_loc,
            samples_per_pixel: 100, // Set the number of samples per pixel for anti-aliasing
        }
    }

    fn get_ray(&self, i: i32, j: i32) -> Ray {
        // Generate a ray from the camera to a specific pixel (i, j)
        let pixel_center =
            &self.pixel00_loc + (i as f64) * &self.pixel_delta_lr + (j as f64) * &self.pixel_delta_ud;
        
        let pixel_sample = pixel_center + self.pixel_sample_square();

        let ray_origin = self.camera_center; // The origin of the ray is the camera center
        let ray_dir = &pixel_sample - ray_origin; // Calculate the direction of the ray
        Ray {
            origin: self.camera_center,
            direction: ray_dir,
        }
    }

    fn pixel_sample_square(&self) -> DVec3 {
        // Generate a random sample within the pixel square for anti-aliasing
        let mut rng = rand::rng(); // Create a random number generator
        let px = -0.5 + rng.random::<f64>(); // Generate a random offset in the x direction
        let py = -0.5 + rng.random::<f64>(); // Generate a random offset in the y direction
        
        (px * self.pixel_delta_lr) + (py * self.pixel_delta_ud) // Calculate the sample position within the pixel square
    }

    fn render(&self, world: &HittableList) -> Image {
        // Render the scene by generating an image based on the camera parameters and the world objects
        Image::new_with_init(self.img_height, self.img_width, |row, col| {
            // Initialize each pixel with a color based on its position
            /*
            let pixel_center =
                &self.pixel00_loc + (row as f64) * &self.pixel_delta_ud + (col as f64) * &self.pixel_delta_lr;
            let ray_dir = &pixel_center - &self.camera_center; // Calculate the direction of the ray from the camera center to the pixel center
            let ray = Ray {
                origin: self.camera_center,
                direction: ray_dir,
            }; // Create a new ray with the camera center as the origin and the calculated direction
            */

            let scale_factor = (self.samples_per_pixel as f64).recip(); // Calculate the scale factor for averaging colors
            let multisampled_pixel_color = (0..self.samples_per_pixel)
                .into_iter() // Create an iterator for the number of samples per pixel
                .map(|_| {
                    let ray = self.get_ray(col as i32, row as i32); // Get a ray for the current pixel
                    ray.color(world) * self.max_value // Calculate the color of the ray based on its intersection with objects in the world
                })
                .fold(DVec3::ZERO, |acc, color| acc + color) * scale_factor; // Sum the colors and apply the scale factor

            //let pixel_color = ray.color(world) * self.max_value; // Calculate the color of the pixel based on the ray's color and scale it to max_value
            Pixel {
                r: multisampled_pixel_color.x as u8, // Extract the red component of the color
                g: multisampled_pixel_color.y as u8, // Extract the green component of the color
                b: multisampled_pixel_color.z as u8, // Extract the blue component of the color
            }
        })
    }
}

fn basic_scene() -> Image {
    // Create a basic scene with a camera and some hittable objects

    let camera = Camera::new(400, 16.0 / 9.0); // Create a new camera with a width of 400 pixels and an aspect ratio of 16:9

    // Create a world with hittable objects
    let mut world = HittableList { objects: vec![] }; // Initialize a new HittableList to hold the objects in the scene

    world.add(Sphere {
        // Create a sphere object
        center: DVec3::new(0.0, 0.0, -1.0),
        radius: 0.5,
    });
    world.add(Sphere {
        center: DVec3::new(0., -100.5, -1.), // Create a ground sphere object
        radius: 100.,
    });

    // Render the scene using the camera and the world objects
    camera.render(&world) // Return the rendered image
    
}

// Trait to define hittable objects
trait Hittable {
    // Define a trait for objects that can be hit by rays
    fn hit(&self, ray: &Ray, interval: StdRange<f64>) -> Option<HitRecord>; // Define a method to check if a ray hits the object within a specified range
}

struct HitRecord {
    // Represents the details of a hit between a ray and an object
    point: DVec3,     // The point of intersection
    normal: DVec3,    // The normal vector at the intersection point
    t: f64,           // The distance along the ray to the intersection point
    front_face: bool, // Indicates if the ray hit the front face of the object
}

impl HitRecord {
    // Create a new HitRecord with the intersection details

    fn with_face_normal(point: DVec3, outward_normal: DVec3, t: f64, ray: &Ray) -> Self {
        // Create a HitRecord with the given point, outward normal, distance t, and ray
        let (front_face, normal) = HitRecord::calc_face_normal(ray, outward_normal); // Calculate the front face and normal vector based on the ray and outward normal
        HitRecord {
            // Create a new HitRecord instance with the calculated values
            point,
            normal,
            t,
            front_face,
        }
    }

    // Function to calculate the front face and normal vector based on the ray and outward normal
    fn calc_face_normal(ray: &Ray, outward_normal: DVec3) -> (bool, DVec3) {
        let front_face = ray.direction.dot(outward_normal) < 0.0; // Check if the ray is hitting the front face
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        }; // Set the normal vector based on whether it's the front face or not
        (front_face, normal) // Return a tuple containing the front face status and the normal vector
    }

    //unused
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: DVec3) {
        let (front_face, normal) = HitRecord::calc_face_normal(ray, outward_normal); // Calculate the front face and normal vector
        self.front_face = front_face; // Set the front face status
        self.normal = normal; // Set the normal vector
    }
}

struct Sphere {
    // Represents a sphere in 3D space
    center: DVec3, // The center of the sphere
    radius: f64,   // The radius of the sphere
}

impl Hittable for Sphere {
    // Implement the Hittable trait for the Sphere struct
    fn hit(&self, ray: &Ray, interval: StdRange<f64>) -> Option<HitRecord> {
        // Check if the ray intersects with the sphere within the specified range
        let origin_center = ray.origin - self.center; // Calculate the vector from the ray's origin to the sphere's center
        let a = ray.direction.length_squared(); // Calculate the dot product of the ray's direction with itself
        let half_b = origin_center.dot(ray.direction); // Calculate the dot product of the origin-center vector with the ray's direction
        let c = origin_center.length_squared() - self.radius * self.radius; // Calculate the dot product of the origin-center vector with itself, minus the square of the radius

        let discriminant = half_b * half_b - a * c; // Calculate the discriminant of the quadratic equation

        if discriminant < 0. {
            // If the discriminant is negative, there is no intersection
            return None; // Return None to indicate no intersection
        }

        let sqrt_d = discriminant.sqrt(); // Calculate the square root of the discriminant
        let mut t = (-half_b - sqrt_d) / a; // (ROOT) Calculate the first intersection point using the quadratic formula

        if !interval.contains(&t) {
            // If the first intersection point is not within the specified range
            t = (-half_b + sqrt_d) / a; // Calculate the second intersection point
            if !interval.contains(&t) {
                // If the second intersection point is also not within the specified range
                return None; // Return None to indicate no intersection
            }
        }

        let point = ray.at(t); // Calculate the intersection point along the ray
        let outward_normal = (point - self.center) / self.radius; // Calculate the outward normal vector at the intersection point

        let rec = HitRecord::with_face_normal(point, outward_normal, t, ray); // Create a HitRecord with the intersection details

        Some(rec) // Return a HitRecord containing the intersection details
    }
}

struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
} // A list of hittable objects

impl HittableList { // Implement methods for the HittableList struct
    fn clear(&mut self) {
        self.objects = vec![] // Clear the list of objects
    }

    fn add<T>(&mut self, object: T) // Add a hittable object to the list
    where
        T: Hittable + 'static, // Ensure that the object implements the Hittable trait and has a static lifetime
    {
        self.objects.push(Box::new(object)); // Add the object to the list by boxing it
    }
}

impl Hittable for HittableList { // Implement the Hittable trait for the HittableList struct
    fn hit(&self, ray: &Ray, interval: StdRange<f64>) -> Option<HitRecord> {
        let (_closest, hit_record) = self.objects.iter().fold((interval.end, None), |acc, item| {
            if let Some(temp_rec) = item.hit(ray, interval.start..acc.0) {
                (temp_rec.t, Some(temp_rec)) // If the item hits, return the hit record and its distance
            } else {
                acc // If the item does not hit, return the accumulator
            }
        });

        hit_record // Return the closest hit record, if any
    }
}

// Function to create a sample image
fn sample_image() -> Image {
    let image_width = 256;
    let image_height = 256;
    Image::new_with_init(image_height, image_width, |row, col| {
        // Initialize each pixel with a color based on its position
        let r = col as f64 / ((image_width - 1) as f64);
        let g = row as f64 / ((image_height - 1) as f64);
        let b = 0.;

        let factor = 255.999; // Scale factor for RGB values

        let r = (r * factor) as u8;
        let g = (g * factor) as u8;
        let b = (b * factor) as u8;
        Pixel { r, g, b } // Create a Pixel with the calculated RGB values
    })
}

#[derive(Default)] // Default implementation for Pixel struct

struct Pixel {
    // Represents a pixel in the image
    // Each pixel has three components: red, green, and blue. In byte values.
    r: u8,
    g: u8,
    b: u8,
}

struct Image {
    // Represents an image made up of pixels
    pixels: Vec<Vec<Pixel>>, //2D vector of pixels
}

// Implementing methods for the Image struct
impl Image {
    pub fn new(height: usize, width: usize) -> Self {
        // Create a new image with specified height and width
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

    pub fn height(&self) -> usize {
        // Return the height of the image
        self.pixels.len()
    }

    pub fn width(&self) -> usize {
        // Return the width of the image
        self.pixels[0].len()
    }
}

struct Ray {
    // Represents a ray in 3D space
    origin: DVec3,
    direction: DVec3,
}

impl Ray {
    // Create a new Ray with a given origin and direction
    fn at(&self, t: f64) -> DVec3 {
        self.origin + t * self.direction
    }

    fn color<T>(&self, world: &T) -> DVec3 // Calculate the color of the ray based on its intersection with objects in the world
    where
        T: Hittable, // Ensure that T implements the Hittable trait
    {
        if let Some(rec) = world.hit(self, 0.0..f64::INFINITY) {
            // Check if the ray hits any object in the world
            return 0.5 * (rec.normal + DVec3::new(1.0, 1.0, 1.0)); // Return a color based on the normal at the hit point
        }

        // If the ray does not hit any object, return a background color
        let unit_direction: DVec3 = self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - a) * DVec3::new(1.0, 1.0, 1.0) + a * DVec3::new(0.5, 0.7, 1.0);
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
