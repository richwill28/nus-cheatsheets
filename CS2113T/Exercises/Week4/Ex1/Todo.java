public class Todo extends Task {
    protected boolean isDone;

    public boolean isDone() {
        return isDone;
    }

    public void setDone(boolean done) {
        isDone = done;
    }

    public Todo(String description) {
        super(description);
        this.isDone = false;
    }
}
