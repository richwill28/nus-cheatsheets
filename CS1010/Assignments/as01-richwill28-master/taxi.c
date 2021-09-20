/**
 * CS1010 Semester 1 AY20/21
 * Assignment 1: Taxi
 *
 * Read in four positive integer corresponding to the day of the week,
 * the hour and minute of boarding, and the distance travelled.  Print
 * to the standard output, the fare of the taxi.
 *
 * @file: taxi.c
 * @author: Richard Willie (Group C08)
 */

#include "cs1010.h"
#include <stdbool.h>

double taxi_fare(long distance)
{
  double value = 3.2;
  // for distance < 1000

  if ((distance > 1000) && (distance <= 10000))
  {
    if ((distance - 1000) % 400 == 0)
    {
      value = 3.2 + (((distance - 1000) / 400) * 0.22);
    }
    else
    {
      // ceiling function
      value = 3.2 + (((distance - 1000) / 400 + 1) * 0.22);
    }
  }
  else if (distance > 10000)
  {
    if ((distance - 10000) % 350 == 0)
    {
      value = 3.2 + (23 * 0.22) + (((distance - 10000) / 350) * 0.22);
    }
    else
    {
      // ceiling function
      value = 3.2 + (23 * 0.22) + (((distance - 10000) / 350 + 1) * 0.22);
    }
  }

  return value;
}

bool is_weekday(long day)
{
  if ((day >= 1) && (day <= 5))
  {
    return true;
  }
  return false;
}

bool is_morning_peak(long day, long hour, long minute)
{
  if (is_weekday(day))
  {
    if (((hour >= 6) && (hour <= 8)) || ((hour == 9) && (minute <= 29)))
    {
      return true;
    }
  }
  return false;
}

bool is_evening_peak(long hour)
{
  if ((hour >= 18) && (hour <= 23))
  {
    return true;
  }
  return false;
}

bool is_midnight_peak(long hour)
{
  if ((hour >= 0) && (hour <= 5))
  { 
    return true;
  }
  return false;
}

double surcharge(long day, long hour, long minute, double fare)
{
  double charge = 0;

  if (is_morning_peak(day, hour, minute))
  {
    charge = 0.25 * fare;
  }
  else if (is_evening_peak(hour))
  {
    charge = 0.25 * fare;
  }
  else if (is_midnight_peak(hour))
  {
    charge = 0.5 * fare;
  }

  return charge;
}

int main()
{
  long day = cs1010_read_long();
  long hour = cs1010_read_long();
  long minute = cs1010_read_long();
  long distance = cs1010_read_long();

  double fare = taxi_fare(distance);
  fare += surcharge(day, hour, minute, fare);

  cs1010_println_double(fare);

  return 0;
}
