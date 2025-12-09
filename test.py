contents = '''Based on the given: activity pattern, the persona could be described as a person with a busy schedule that involves significant daily travel. They have a mix of weekday and weekend routines, with different starting and ending times for their trips.
During weekdays, they travel over 50 kilometers a day, starting at 06:50 and ending at 21:20. They visit Convenience Store#358 at the beginning of the day and Convenience Store#10197 before returning home. This suggests that they may be involved in some form of daily errands or routine that requires frequent stops at these stores.'''
for c in contents.split("\n"):
    if c:
        role, description = c.split(": ")
        print(f"Role: {role}, Description: {description}")
        break