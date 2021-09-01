/open Array.java
Integer i
String s
Array<Integer> a;
a = new Array<Integer>(4);
a.set(0, 3);
a.set(1, 6);
a.set(2, 4);
a.set(3, 1);
a.set(0, "huat");
i = a.get(0)
i
i = a.get(1)
i
i = a.get(2)
i
i = a.get(3)
i
s = a.get(0)
i = a.min()
i
a.set(3,9);
i = a.min()
i
// try something not comparable
class A {}
Array<A> a;
class A implements Comparable<Long> { public int compareTo(Long i) { return 0; } }
Array<A> a;
// try something comparable
class A implements Comparable<A> { public int compareTo(A a) { return 0; } }
Array<A> a;
