import ssl
import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
import ipaddress
import subprocess

def generate_self_signed_cert():
    """Generate self-signed SSL certificate for local development"""
    
    # Create certificates directory
    cert_dir = "ssl_certs"
    if not os.path.exists(cert_dir):
        os.makedirs(cert_dir)
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Get local IP address
    try:
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
        lines = result.stdout.split('\n')
        local_ip = None
        for line in lines:
            if 'IPv4' in line and '192.168' in line:
                local_ip = line.split(':')[-1].strip()
                break
        if not local_ip:
            local_ip = "192.168.1.6"  # fallback
    except:
        local_ip = "192.168.1.6"  # fallback
    
    print(f"🔐 Generating certificate for IP: {local_ip}")
    
    # Certificate subject
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CortexMD Dev"),
        x509.NameAttribute(NameOID.COMMON_NAME, local_ip),
    ])
    
    # Certificate valid for 1 year
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("127.0.0.1"),
            x509.IPAddress(ipaddress.ip_address(local_ip)),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256())
    
    # Write certificate to file
    cert_path = os.path.join(cert_dir, "cert.pem")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    # Write private key to file
    key_path = os.path.join(cert_dir, "key.pem")
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    print(f"✅ SSL Certificate generated:")
    print(f"   📄 Certificate: {cert_path}")
    print(f"   🔑 Private Key: {key_path}")
    print(f"   🌐 Valid for: {local_ip}, localhost, 127.0.0.1")
    
    return cert_path, key_path, local_ip

if __name__ == "__main__":
    generate_self_signed_cert()
