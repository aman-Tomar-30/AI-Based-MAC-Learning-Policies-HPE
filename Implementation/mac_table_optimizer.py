import topo 
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
#print(r.ping()) 

def get_mac_table(switch):
    return topo.print_fdb(switch)

mac_table = get_mac_table(switch="s1") # {mac1: entry1, mac2: entry2...}
#print(mac_table)

def insert_into_hash(mac_table):
    for mac,value in mac_table.items():
        r.hset(mac, mapping=value)
    
    print(f"Insertion Complete, total values: {len(mac_table)}")

insert_into_hash(mac_table)

def get_mac_age_info(mac_table):
    mac_age = {}
    for mac,value in mac_table.items():
        #print(mac, value['age'])
        age = int(value["age"])
        mac_age[mac] = age

    #print(mac_age) # {mac1:age1, mac2:age2}
    return mac_age

mac_age = get_mac_age_info(mac_table)

key = "mac_index"

def insert_into_zset(mac_age):
    for mac,age in mac_age.items():
        r.zadd(key, {mac:age})

#delete existing entries from zset before insertion 
r.delete(key)

print("Before insert:", r.zcard(key))
insert_into_zset(mac_age)
print("After insert:", r.zcard(key))

#print(r.zrange(key, 0 ,-1, withscores=True)) #gives mac with ascending order to their ages

def age_based_removal(start=60, end=300):
    print(f"Running deletion based on aging time, entries having more than {start} sec will be removed")
    r.zremrangebyscore(key, start, end) # delete entries with age between 60 and 300

    print(r.zrange(key, 0 ,-1, withscores=True)) # after removing

    remaining_entries = r.zrange(key, 0, -1, withscores=True)

    remaining_macs = {entry[0] for entry in remaining_entries} # get macs of remaining entries

    for mac in mac_age: #delete other than remaining macs from hash 
        if mac not in remaining_macs:
            r.delete(mac) 

    # Show remaining data
    for mac in remaining_macs:
        print(r.hgetall(mac))

def size_based_removal(size=5):
    print(f"Running deletion based on size only {size} entries will be remaining")
    r.zremrangebyrank(key, size, -1) # delete entries after 5th rank 

    print(r.zrange(key, 0 ,-1, withscores=True)) # after removing

    remaining_entries = r.zrange(key, 0, -1, withscores=True)
    remaining_macs = {entry[0] for entry in remaining_entries} # get macs of remaining entries

    for mac in mac_age: #delete other than remaining macs from hash 
        if mac not in remaining_macs:
            r.delete(mac) 

    # Show remaining data
    for mac in remaining_macs:
        print(r.hgetall(mac))

size_based_removal() #only top 5 entries remain
age_based_removal() # only entries with less than 60 Sec age remain