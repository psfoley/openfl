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

SAN=$subject_alt_name openssl req -new -config config/client.conf -out $common_name.csr -keyout $common_name.key -subj "/CN=$common_name /WD=123456"
openssl ca -config config/signing-ca.conf -batch -in $common_name.csr -out $common_name.crt

mkdir -p $common_name
mv $common_name.crt $common_name.key $common_name
rm $common_name.csr
