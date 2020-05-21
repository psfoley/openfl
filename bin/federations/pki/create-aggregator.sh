# ensure we are in the pki directory
cd $(dirname $0)

# default common_name to hostname.domainname
common_name=$(hostname).$(hostname -d)
subject_alt_name=DNS:$common_name

while getopts ":c:s:i:" opt; do
  case $opt in
    c) common_name="$OPTARG"
    ;;
    s) subject_alt_name=$subject_alt_name,DNS:"$OPTARG"
    ;;
    i) subject_alt_name=$subject_alt_name,IP:$OPTARG
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo $subject_alt_name

SAN=$subject_alt_name openssl req -new -config config/server.conf -subj "/CN=$common_name" -out $common_name.csr -keyout $common_name.key
openssl ca -config config/signing-ca.conf -batch -extensions server_ext -in $common_name.csr -out $common_name.crt

filename_base=agg_$common_name

mkdir -p $filename_base
mv $common_name.crt $filename_base/$filename_base.crt
mv $common_name.key $filename_base/$filename_base.key
rm $common_name.csr
