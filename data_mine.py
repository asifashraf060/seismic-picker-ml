import sqlite3
import pickle
import numpy as np
from obspy import UTCDateTime, read
from obspy.clients.fdsn.client import Client
from obspy.core.inventory.inventory import Inventory
import seisbench.models as sbm
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from io import BytesIO
from tqdm import tqdm
import os

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ S3 CONFIGURATION FOR NCEDC DATA ACCESS
# ═══════════════════════════════════════════════════════════════════════════════════════

# Configure S3 Client for Public NCEDC Access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-west-2')
BUCKET_NAME = 'ncedc-pds'

# Load pretrained PhaseNet for P-wave picks
picker = sbm.PhaseNet.from_pretrained("original")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def station_data_exists(station, eq_time, pre_time, post_time, client: Client, 
                       network: str, location: str, channel: str,
                       s3_client, bucket_name: str) -> bool:
    """Check if station data exists in S3"""
    day0 = eq_time.replace(hour=0, minute=0, second=0, microsecond=0)
    jul = day0.julday
    fname = f"{station.code}.{network}.{channel}..D.{day0.year}.{jul:03d}"
    key = f"continuous_waveforms/{network}/{day0.year}/{day0.year}.{jul:03d}/{fname}"
    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError:
        return False

def filter_inventory(inventory: Inventory, eq_time, pre_time, post_time, 
                    client: Client, network: str, location: str, channel: str,
                    s3_client, bucket_name: str) -> Inventory:
    """Filter inventory to only include stations with available data"""
    kept_networks = []
    for net in inventory.networks:
        kept_stns = []
        for st in net.stations:
            if station_data_exists(st, eq_time, pre_time, post_time, client, 
                                 network, location, channel, s3_client, bucket_name):
                kept_stns.append(st)
        if kept_stns:
            net.stations = kept_stns
            kept_networks.append(net)
    
    inventory.networks = kept_networks
    return inventory

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ DATABASE CREATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_database(db_path='seismic_data.db'):
    """Create SQLite database with required schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create earthquakes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS earthquakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time REAL,
            latitude REAL,
            longitude REAL,
            magnitude REAL,
            depth REAL
        )
    ''')
    
    # Create waveforms table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS waveforms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            earthquake_id INTEGER,
            station_code TEXT,
            network TEXT,
            channel TEXT,
            location TEXT,
            waveform_data BLOB,
            sampling_rate REAL,
            p_pick_time REAL,
            eq_time REAL,
            pre_time REAL,
            post_time REAL,
            distance_km REAL,
            FOREIGN KEY (earthquake_id) REFERENCES earthquakes (id)
        )
    ''')
    
    conn.commit()
    return conn

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════════════

def process_station(station, start_time, pre_time, post_time, eq_time,
                   network, channel, inventory):
    """Process individual station and return waveform data"""
    station_code = station.code
    
    # Build S3 key
    file_name = f'{station_code}.{network}.{channel}..D.{start_time.year}.{start_time.julday:03d}'
    key = f"continuous_waveforms/{network}/{start_time.year}/{start_time.year}.{start_time.julday:03d}/{file_name}"
    
    try:
        # Stream data from S3
        resp = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data_stream = resp['Body']
        buff = BytesIO(data_stream.read())
        buff.seek(0)
        
        # Read waveform
        st_stream = read(buff, format='MSEED')
        st_stream.trim(starttime=eq_time - pre_time, endtime=eq_time + post_time)
        
        if len(st_stream) < 1:
            print(f"  Skipping station {station_code} (no data)")
            return None
        
        # Get single trace
        tr = st_stream[0]
        
        # Remove instrument response
        try:
            tr.remove_response(inventory=inventory, output="DISP", zero_mean=True)
        except Exception as e:
            print(f"  Warning: Could not remove response for {station_code}: {e}")
        
        # Get PhaseNet picks for reference
        picks = picker.classify(st_stream, batch_size=256, P_threshold=0.075, S_threshold=0.1).picks
        if not picks:
            print(f"  No picks found for station {station_code}")
            return None
        
        # Use first P arrival
        p_time = picks[0].peak_time
        
        # Calculate distance
        station_lat = station.latitude
        station_lon = station.longitude
        
        # Return the processed data
        return {
            'station_code': station_code,
            'waveform': tr.data,
            'sampling_rate': tr.stats.sampling_rate,
            'p_pick_time': p_time.timestamp,
            'eq_time': eq_time.timestamp,
            'pre_time': pre_time,
            'post_time': post_time,
            'station_lat': station_lat,
            'station_lon': station_lon
        }
        
    except Exception as e:
        print(f"  Error processing station {station_code}: {e}")
        return None

def mine_earthquake_data(eq_time, eq_lat, eq_lon, eq_mag=None, eq_depth=None,
                        radius_km=250, network='NC', channel='HNE', 
                        pre_time=3, post_time=120):
    """Mine seismic data for a single earthquake"""
    
    if not isinstance(eq_time, UTCDateTime):
        eq_time = UTCDateTime(eq_time)
    
    # Define time window
    start_time = eq_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = eq_time.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Get station inventory
    client = Client('NCEDC')
    print(f"\nProcessing earthquake at {eq_time}")
    print("Getting station inventory...")
    
    inventory = client.get_stations(
        network=network, latitude=eq_lat, longitude=eq_lon,
        starttime=start_time, endtime=end_time, maxradius=radius_km/111.2,
        location='*', channel=channel, level="response"
    )
    
    print("Filtering inventory for available data...")
    inventory = filter_inventory(inventory, eq_time, pre_time, post_time, 
                               client, network, '*', channel, s3, BUCKET_NAME)
    
    stations = inventory[0].stations if inventory else []
    print(f"Found {len(stations)} stations with data")
    
    # Process each station
    waveforms = []
    for station in tqdm(stations, desc="Processing stations"):
        result = process_station(station, start_time, pre_time, post_time, 
                               eq_time, network, channel, inventory)
        if result:
            # Calculate distance from earthquake
            dist_deg = np.sqrt((station.latitude - eq_lat)**2 + 
                              (station.longitude - eq_lon)**2)
            result['distance_km'] = dist_deg * 111.2
            result['network'] = network
            result['channel'] = channel
            result['location'] = ''
            result['eq_lat'] = eq_lat
            result['eq_lon'] = eq_lon
            result['eq_mag'] = eq_mag
            result['eq_depth'] = eq_depth
            waveforms.append(result)
    
    print(f"Successfully processed {len(waveforms)} waveforms")
    return waveforms

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN DATA MINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def main():
    """Main data mining function"""
    
    print("="*80)
    print("SEISMIC DATA MINING FOR ML PIPELINE")
    print("="*80)
    
    # Create database
    db_path = 'seismic_data.db'
    print(f"\nCreating database: {db_path}")
    conn = create_database(db_path)
    cursor = conn.cursor()
    
    # Define earthquakes to process
    earthquakes = [
        {
            "time": '2022-12-20T10:34:24', 
            "lat": 40.369, 
            "lon": -124.588, 
            "mag": 6.4,
            "depth": 10.0,
            "radius": 250,
            "name": "Ferndale"
        },
        {
            "time": "2010-01-10T00:27:39", 
            "lat": 40.652, 
            "lon": -124.693, 
            "mag": 6.5,
            "depth": 15.0,
            "radius": 250,
            "name": "Eureka"
        },
        {
            "time": "2021-12-20T20:10:31", 
            "lat": 40.390, 
            "lon": -124.298, 
            "mag": 5.8,
            "depth": 12.0,
            "radius": 250,
            "name": "Petrolia"
        },
        {
            "time": "2021-07-08T22:49:48", 
            "lat": 38.508, 
            "lon": -119.500, 
            "mag": 6.0,
            "depth": 8.0,
            "radius": 250,
            "name": "Antelope"
        },
        {
            "time": "2020-04-11T14:36:37", 
            "lat": 38.053, 
            "lon": -118.733, 
            "mag": 5.5,
            "depth": 10.0,
            "radius": 250,
            "name": "Bodie"
        },
    ]
    
    # Process each earthquake
    total_waveforms = 0
    
    for eq in earthquakes:
        print(f"\n{'='*60}")
        print(f"Processing {eq['name']} earthquake")
        print(f"{'='*60}")
        
        try:
            # Insert earthquake record
            eq_time = UTCDateTime(eq['time'])
            cursor.execute('''
                INSERT INTO earthquakes (event_time, latitude, longitude, magnitude, depth)
                VALUES (?, ?, ?, ?, ?)
            ''', (eq_time.timestamp, eq['lat'], eq['lon'], eq.get('mag'), eq.get('depth')))
            
            earthquake_id = cursor.lastrowid
            
            # Mine waveform data
            waveforms = mine_earthquake_data(
                eq_time=eq['time'],
                eq_lat=eq['lat'],
                eq_lon=eq['lon'],
                eq_mag=eq.get('mag'),
                eq_depth=eq.get('depth'),
                radius_km=eq['radius'],
                network='NC',
                channel='HNE',
                pre_time=3,
                post_time=120
            )
            
            # Insert waveforms into database
            for wf in waveforms:
                # Serialize waveform data
                waveform_blob = pickle.dumps(wf['waveform'])
                
                cursor.execute('''
                    INSERT INTO waveforms (
                        earthquake_id, station_code, network, channel, location,
                        waveform_data, sampling_rate, p_pick_time, eq_time,
                        pre_time, post_time, distance_km
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    earthquake_id,
                    wf['station_code'],
                    wf['network'],
                    wf['channel'],
                    wf['location'],
                    waveform_blob,
                    wf['sampling_rate'],
                    wf['p_pick_time'],
                    wf['eq_time'],
                    wf['pre_time'],
                    wf['post_time'],
                    wf['distance_km']
                ))
            
            conn.commit()
            total_waveforms += len(waveforms)
            print(f"✅ Added {len(waveforms)} waveforms to database")
            
        except Exception as e:
            print(f"❌ Error processing {eq['name']} earthquake: {e}")
            conn.rollback()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DATA MINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total earthquakes processed: {len(earthquakes)}")
    print(f"Total waveforms in database: {total_waveforms}")
    
    # Verify database contents
    cursor.execute("SELECT COUNT(*) FROM earthquakes")
    eq_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM waveforms")
    wf_count = cursor.fetchone()[0]
    
    print(f"\nDatabase verification:")
    print(f"  Earthquakes: {eq_count}")
    print(f"  Waveforms: {wf_count}")
    
    conn.close()
    print(f"\n✅ Database saved to: {db_path}")

if __name__ == "__main__":
    main()