## \[Key Exercise\] create `Rectangle` objects

Update the code below to create a new `Rectangle` object as described in the 
code comments, to produce the given output.

- The `Rectangle` class is found in the `java.awt package`.
- The parameters you need to supply when creating new `Rectangle` objects are 
`(int x, int y, int width, int height)`.

```java
public class Main {
    public static void main(String[] args) {
        Rectangle r;

        // TODO: create a Rectangle object that has the
        // properties x=0, y=0, width=5, height=10
        // and assign it to r
        System.out.println(r);
    }
}
```

⬇️

```console
java.awt.Rectangle[x=0,y=0,width=5,height=10]
```

<details>
  <summary>Hint</summary>

  <ul>
    <li>
      Import the <code>java.awt.Rectangle</code> class
    </li>
    <li>
      This is how you create the required object 
      <code>new Rectangle(0, 0, 5, 10)</code>
    </li>
  </ul>
</details>

Website: https://nus-cs2113-ay2122s1.github.io/website/schedule/week3/topics.html#key-exercise-create-rectangle-objects
