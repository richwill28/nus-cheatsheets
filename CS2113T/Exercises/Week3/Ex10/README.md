## \[Key Exercise\] find total expenditure

Write a program to ask the user for a description of overseas expenses 
(presumably, the user has just returned from an overseas trip) and calculate 
the total in local currency.

- The conversion rate from overseas currency to local currency : overseas `$1.0` = local `$1.70`
- The user can describe expenses is in free form text, as one line. The program takes all amounts mentioned in the format `$amount` e.g., `$1.50`

Here is one example output: ⬇️

```console
Your expenses while overseas?beer $4.50 books $3.00 $5.00 for food, that's all
Expenses in overseas currency:[$4.50, $3.00, $5.00]
Total in local currency: $21.25
```

Here is another: ⬇️

```console
Your expenses while overseas?nothing. I lived off my friends all the time.
Expenses in overseas currency:[]
Total in local currency: $0.00
```

One more: ⬇️

```console
Your expenses while overseas? Just $10
Expenses in overseas currency:[$10]
Total in local currency: $17.00
```

Here's the skeleton code to use as the starting point:

```java
public class Main {

    // You can add more methods here

    public static void main(String[] args) {
        String line;
        Scanner in = new Scanner(System.in);

        System.out.print("Your expenses while overseas?");
       // TODO: add your code here
    }
}
```

💡 You can use the `split` method of the `String` class to convert a sentence into an array of words. e.g.,

```java
String sentence = "hello my dear";

// split using the space as the delimiter
String[] words = sentence.split(" ");

System.out.println(Arrays.toString(words));
```

➡️ `[hello, my, dear]`

<details>
  <summary>Hint</summary>

  💡 You can use `String.format("%.2f", doubleValue)` to format `doubleValue` to two decimal points.
  
  e.g., `String.format("%.2f", 1.3334)` ➡️ `1.33`
</details>

<details>
  <summary>Partial solution</summary>

  ```java
  import java.util.Arrays;
  import java.util.Scanner;

  public class Main {
      public static String[] filterAmounts(String sentence) {
          String[] words = sentence.split(" ");
          String[] result = new String[words.length];
          int wordCount = 0;
          for (String word : words) {
              if (word.startsWith("$")) {
                  result[wordCount] = word;
                  wordCount++;
              }
          }
          return Arrays.copyOf(result, wordCount);
      }

      public static void main(String[] args) {
          String line;
          Scanner in = new Scanner(System.in);

          System.out.print("Your expenses while overseas?");
          line = in.nextLine();

          String[] amounts = filterAmounts(line);
          System.out.println("Expenses in overseas currency:" + Arrays.toString(amounts));
          double total = 0;
          for (String amount : amounts) {
              // convert amount to double, multiply by currency conversion rate, and add to total
          }
          System.out.println("Total in local currency: $" + String.format("%.2f", total));
      }
  }
  ```
</details>

Website: https://nus-cs2113-ay2122s1.github.io/website/schedule/week3/topics.html#key-exercise-find-total-expenditure
