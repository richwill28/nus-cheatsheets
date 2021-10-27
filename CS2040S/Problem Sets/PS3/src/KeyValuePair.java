public class KeyValuePair implements Comparable<KeyValuePair> {
    private int key;
    private int value;

    KeyValuePair(int k, int v) {
        key = k;
        value = v;
    }

    public int getKey() {
        return key;
    }

    public int getValue() {
        return value;
    }

    public int compareTo(KeyValuePair other) {
        if (this.key < other.key) {
            return -1;
        } else if (this.key > other.key) {
            return 1;
        } else {
            return 0;
        }
    }

    public String toString() {
        return "(" + key + ", " + value + ")";
    }
}
