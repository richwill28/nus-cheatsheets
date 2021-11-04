#ifndef Customer_h
#define Customer_h

class Customer {
    private:
        int arrival_time; // time of arrival after the shop opened in min
        int processing_time; // amount of time need to be processed with the customer serivce in min
    public:
        Customer () { arrival_time = 0; processing_time = 0; }
        void setAT(int t) { arrival_time = t; }
        void setPT(int t) { processing_time = t; }
        int AT() { return arrival_time; }
        int PT() { return processing_time; }
        bool operator>(const Customer& c); // a customer is "greater" if its time is shorter
        bool operator<(const Customer& c);
        bool operator==(const Customer& c);
};

void customerQueueTest(int);

#endif
