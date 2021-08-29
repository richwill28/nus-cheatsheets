## \[Key Exercise\] Compare Names

Write a Java program that takes two command line arguments and prints true or 
false to indicate if the two arguments have the same value. Follow the sample 
output given below.

```java
class Main {
    public static void main(String[] args) {
        // add code here
    }
}
```

`java Main adam eve` ⬇️

```console
Words given: adam, eve
They are the same: false
```

`java Main eve eve` ⬇️

```console
Words given: eve, eve
They are the same: true
```

💡 Use the following technique to compare two `Strings` (i.e., don't use `==`). 
Reason: to be covered in a later topic.

```java
String x = "foo";
boolean isSame = x.equals("bar") // false
isSame = x.equals("foo") // true
```

<details>
  <summary>Hint</summary>

  <ul>
    <li>
      The two command line arguments can be accessed inside the 
      <code>main</code> method using <code>args[0]</code> and 
      <code>args[1]</code>.
    </li>
    <li>
      When using multiple operators in the same expression, you might need to 
      use parentheses to specify operator precedence. e.g., 
      <code>"foo" + x == y</code> vs <code>"foo" + (x == y)</code>.
    </li>
  </ul>
</details>

<details>
  <summary>Partial solution</summary>
  
  ```java
  class Main {
      public static void main(String[] args) {
          String first = args[0];
          String second = args[1];
          System.out.println("Words given: " + first + ", " + second);
          // ...
      }
  }
  ```
</details>

Website: https://nus-cs2113-ay2122s1.github.io/website/schedule/week2/topics.html#key-exercise-compare-names
