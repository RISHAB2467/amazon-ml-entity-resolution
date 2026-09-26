import duckdb

c = duckdb.connect("output/profile/profile.duckdb", read_only=True)

print("TEST S1 FRANCE")
print(c.sql("""
SELECT business_name, business_address
FROM test_s1
WHERE country = 'France'
USING SAMPLE 25
"""))

print("TEST S2 FRANCE")
print(c.sql("""
SELECT business_name, business_address
FROM test_s2
WHERE country = 'France'
USING SAMPLE 25
"""))

c.close()
