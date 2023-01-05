## \[Key Exercise\] print `Printable` items

The `Main` class below passes a list of `Printable` objects (i.e., objects that 
implement the `Printable` interface) for another method to be printed.

```java
public class Main {
    public static void printObjects(Printable[] items) {
        for (Printable p : items) {
            p.print();
        }
    }

    public static void main(String[] args) {
        Printable[] printableItems = new Printable[] {
                new Circle(5),
                new Rectangle(3, 4),
                new Person("James Cook")
        };

        printObjects(printableItems);
    }
}
```

⬇️

```console
Circle of area 78
Rectangle of area 12
Person of name James Cook
```

Classes `Shape`, `Circle`, and `Rectangle` are given below:

```java
public abstract class Shape {
    public abstract int area();
}
```

```java
public class Circle extends Shape implements Printable {
    private int radius;

    public Circle(int radius) {
        this.radius = radius;
    }

    @Override
    public int area() {
        return (int)(Math.PI * radius * radius);
    }

    @Override
    public void print() {
        System.out.println("Circle of area " + area());
    }
}
```

```java
public class Rectangle extends Shape implements Printable {
    private int height;
    private int width;

    public Rectangle(int height, int width) {
        this.height = height;
        this.width = width;
    }

    @Override
    public int area() {
        return height * width;
    }

    @Override
    public void print() {
        System.out.println("Rectangle of area " + area());
    }
}
```

Add the missing `Printable` interface. Add the missing methods of the `Person` 
class given below.

```java
public class Person implements Printable {
    private String name;

    // todo: add missing methods
}
```

<details>
  <summary>Partial solution</summary>

  ```java
  public interface Printable {
      //...
  }
  ```
</details>

Website: https://nus-cs2113-ay2122s1.github.io/website/schedule/week5/topics.html#key-exercise-print-printable-items
